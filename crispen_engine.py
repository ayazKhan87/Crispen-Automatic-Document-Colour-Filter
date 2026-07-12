"""
crispen_engine.py
=================

Core engine for **Crispen**: an automatic document clean-up pipeline that takes
a photographed / scanned page and produces a crisp, white-background copy while
keeping any genuine colour (logos, stamps, buttons, photos) untouched.

This module contains ONLY the image-processing building blocks. There is no
web/UI code here - it is meant to be imported by an API layer (see app.py).

Public entry point:
    CrispenEngine(...).run(bgr_image) -> (cleaned_bgr, stage_gallery)

Every helper is named around the auto-doc-clean use case. The underlying maths
(LAB/HSV colour work, guided filtering, adaptive thresholds, background
bleaching) is the same tried-and-tested logic - only the vocabulary changed.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Numeric safety helpers
# ---------------------------------------------------------------------------

def guarded_ratio(top, bottom, epsilon=1e-10, default=0.0):
    """
    Divide two arrays without ever emitting NaN/Inf.

    A tiny denominator is clamped to `epsilon` first, and any result that still
    turns out non-finite is swapped for `default`. Used everywhere ratios are
    computed on paper/ink pixels so a stray zero never poisons the pipeline.
    """
    bottom_safe = np.where(
        np.abs(bottom) < epsilon,
        epsilon,
        bottom
    )
    outcome = top / bottom_safe
    # Any leftover NaN/Inf gets replaced with the fallback value.
    outcome = np.where(np.isfinite(outcome), outcome, default)
    return outcome


def scrub_invalid_values(arr, fill_value=0):
    """
    Replace every NaN and Inf in `arr` with `fill_value`, keeping the dtype.

    A cheap defensive wipe applied after risky operations (sqrt, division,
    guided filtering) before the mask/frame is used downstream.
    """
    # Fast path: a scalar fill lets np.nan_to_num do NaN/+Inf/-Inf in one C
    # pass instead of two np.where allocations. Falls back to the general
    # broadcasting form when the caller passes a per-pixel fill array.
    if np.isscalar(fill_value):
        return np.nan_to_num(
            arr,
            nan=float(fill_value),
            posinf=float(fill_value),
            neginf=float(fill_value)
        )
    arr = np.where(np.isnan(arr), fill_value, arr)
    arr = np.where(np.isinf(arr), fill_value, arr)
    return arr


def _scale_blur_window(image_shape, base_ratio=0.06, min_size=51):
    """
    Pick an odd blur/kernel width that grows with the page's diagonal.

    High-resolution scans get proportionally wider kernels so transitions stay
    smooth instead of looking pixel-locked on big images.
    """
    if len(image_shape) < 2:
        return min_size | 1
    height, width = image_shape[:2]
    diagonal = np.sqrt(height * height + width * width)
    window = int(max(min_size, diagonal * base_ratio))
    if window % 2 == 0:
        window += 1
    return window


def _soft_ramp(low_edge, high_edge, x):
    """Smooth Hermite ramp from 0 to 1 between two edges (no hard cut-offs)."""
    # Guard the span so edges that nearly coincide don't blow up.
    span = high_edge - low_edge
    span = np.where(np.abs(span) < 1e-8, 1.0, span)
    x = guarded_ratio(x - low_edge, span, epsilon=1e-8, default=0.0)
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _logistic_gate(x, midpoint=0.5, steepness=10.0):
    """Logistic curve tuned for soft-mask shaping."""
    return 1.0 / (1.0 + np.exp(-steepness * (x - midpoint)))


def _boundary_preserving_blur(mask, guide_image, radius_ratio=0.05, eps=1e-3):
    """
    Smooth a soft mask while keeping it glued to real edges via a guided filter.

    Falls back to a plain Gaussian blur when OpenCV's ximgproc module is not
    installed, so the pipeline still runs on stripped-down builds.
    """
    radius = int(max(15, min(guide_image.shape[:2]) * radius_ratio))
    if radius % 2 == 0:
        radius += 1
    try:
        # The guide must be a single-channel uint8 image.
        if len(guide_image.shape) == 3:
            guide_single = cv2.cvtColor((guide_image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            guide_single = (guide_image * 255).astype(np.uint8) if guide_image.max() <= 1.0 else guide_image.astype(np.uint8)

        mask_uint8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)

        guided = cv2.ximgproc.guidedFilter(
            guide=guide_single,
            src=mask_uint8,
            radius=radius,
            eps=eps
        )
        return guided.astype(np.float32) / 255.0
    except (AttributeError, cv2.error):
        # No guided filter available - use a wide Gaussian instead.
        kernel_size = radius if radius % 2 == 1 else radius + 1
        return cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)


def _tile_scan_regions(image_shape, window_size=256, stride=128):
    """Produce overlapping tile coordinates for sliding-window paper analysis."""
    height, width = image_shape[:2]
    tiles = []
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            y_end = min(y + window_size, height)
            x_end = min(x + window_size, width)
            tiles.append((y, y_end, x, x_end))
    return tiles


def _measure_processing_defects(source, cleaned):
    """Score common clean-up side effects so they can be corrected adaptively."""
    src_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    out_gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    src_texture = cv2.Laplacian(src_gray, cv2.CV_32F).var() + 1e-6
    out_texture = cv2.Laplacian(out_gray, cv2.CV_32F).var() + 1e-6
    texture_loss_val = guarded_ratio(src_texture - out_texture, src_texture, epsilon=1e-6, default=0.0)
    texture_loss = max(0.0, texture_loss_val)

    diff = out_gray - src_gray
    low_freq = cv2.GaussianBlur(diff, (31, 31), 0)
    patchiness = np.std(low_freq)

    tone_shift = np.abs(np.mean(out_gray) - np.mean(src_gray))

    return {
        'texture_loss': texture_loss,
        'patchiness': patchiness,
        'hist_shift': tone_shift
    }


def _repair_processing_defects(cleaned, metrics, content_protection_mask=None):
    """
    Apply gentle corrective passes when defect scores cross safe thresholds.

    CRITICAL: only ever touches background pixels - every content region stays
    exactly as delivered by the main pipeline.
    """
    result = cleaned.astype(np.float32)
    baseline = cleaned.astype(np.float32)

    # Derive a content-protection mask if the caller didn't supply one.
    if content_protection_mask is None:
        gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY).astype(np.float32)
        norm_gray = gray / 255.0

        # Content boundaries show up as edges.
        edges = cv2.Canny(gray.astype(np.uint8), 40, 120)
        edges_dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        has_edges = (edges_dilated > 0).astype(np.float32)

        # Dark ink / photo regions also count as content.
        is_dark = (norm_gray < 0.5).astype(np.float32)

        content_protection_mask = np.maximum(has_edges, is_dark)
        try:
            content_protection_mask = _boundary_preserving_blur(content_protection_mask, norm_gray, radius_ratio=0.02, eps=1e-3)
        except Exception:
            kernel_size = _scale_blur_window(cleaned.shape, base_ratio=0.02, min_size=9)
            content_protection_mask = cv2.GaussianBlur(content_protection_mask, (kernel_size, kernel_size), 0)
        content_protection_mask = np.clip(content_protection_mask, 0.0, 1.0)

    if content_protection_mask.max() > 1.0:
        content_protection_mask = content_protection_mask / 255.0
    content_protection_mask = np.clip(content_protection_mask, 0.0, 1.0)

    # Background = wherever we are allowed to retouch.
    background_mask = 1.0 - content_protection_mask
    background_mask = np.clip(background_mask, 0.0, 1.0)

    background_mask_3ch = cv2.merge([background_mask, background_mask, background_mask])
    content_mask_3ch = cv2.merge([content_protection_mask, content_protection_mask, content_protection_mask])

    # Patchiness -> smooth the background only.
    if metrics['patchiness'] > 0.02:
        filtered = cv2.bilateralFilter(result.astype(np.uint8), 9, 40, 40).astype(np.float32)
        result = result * content_mask_3ch + filtered * background_mask_3ch

    # Lost texture -> re-inject a little sharpness on background only.
    if metrics['texture_loss'] > 0.25:
        lap = cv2.Laplacian(result, cv2.CV_32F)
        sharpened = cv2.addWeighted(result, 1.0, lap, 0.15, 0)
        result = result * content_mask_3ch + sharpened * background_mask_3ch

    # Tone drift -> pull the background back toward the pre-repair tone.
    if metrics['hist_shift'] > 0.08:
        corrected = cv2.addWeighted(result, 0.85, baseline, 0.15, 0)
        result = result * content_mask_3ch + corrected * background_mask_3ch

    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Page category classification
# ---------------------------------------------------------------------------

class PageCategoryClassifier:
    """
    Decide, from the pixels alone, whether a scan is a plain white page or a
    colourful product/packaging shot. Robust to uneven lighting.
    """

    @staticmethod
    def categorize(image, saturation_threshold=40):
        """
        Classify the incoming page.

        Returns:
            tuple: (category, suggested_saturation_threshold)
                category: 'white_document' or 'colored_product'
                suggested_saturation_threshold: tuned per-image cut-off
        """
        # First: is a big (>20%) genuinely multicoloured area present? That
        # flips even a mostly-white page into the "colored" bucket (e.g. a
        # white sheet with a large embedded photo).
        has_big_color_area = PageCategoryClassifier._has_large_color_region(image, min_area_ratio=0.20)

        traits = PageCategoryClassifier._collect_traits(image)
        category = PageCategoryClassifier._decide_category(traits, has_big_color_area)

        suggested_threshold = PageCategoryClassifier._suggest_threshold(traits, category)

        return category, suggested_threshold

    @staticmethod
    def _has_large_color_region(image, min_area_ratio=0.20):
        """
        Report whether a large, genuinely multi-hued region exists (photo/image
        embedded in an otherwise plain page), as opposed to a flat button/stamp.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        h, s, v = cv2.split(hsv)
        l, a, b = cv2.split(lab)

        height, width = image.shape[:2]
        total_pixels = height * width
        min_area_pixels = int(total_pixels * min_area_ratio)

        # Genuine colour = saturated AND not near-black.
        is_colored = (s > 40) & (v > 50)
        colored_mask = is_colored.astype(np.uint8) * 255

        # Bridge neighbouring colour pixels into solid blobs.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        colored_mask = cv2.morphologyEx(colored_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        colored_mask = cv2.morphologyEx(colored_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(colored_mask, connectivity=8)

        for label in range(1, num_labels):  # 0 is background
            region_area = stats[label, cv2.CC_STAT_AREA]

            if region_area < min_area_pixels:
                continue

            region_mask = (labels == label).astype(np.uint8)

            s_region_all = s[region_mask > 0]
            v_region_all = v[region_mask > 0]

            if len(s_region_all) == 0:
                continue

            mean_saturation = np.mean(s_region_all)
            mean_brightness = np.mean(v_region_all)

            # Skip washed-out or near-black blobs (grey shadow, black text).
            if mean_saturation < 50 or mean_brightness < 60:
                continue

            # Multi-hued photos vary a lot in LAB; flat buttons barely vary.
            a_region = a[region_mask > 0]
            b_region = b[region_mask > 0]

            if len(a_region) == 0:
                continue

            a_variance = np.var(a_region)
            b_variance = np.var(b_region)
            total_color_variance = a_variance + b_variance
            is_multicolored = total_color_variance > 200

            # Photos also have spread-out saturation.
            s_variance = np.var(s_region_all)
            has_varied_saturation = s_variance > 500

            # ...and lightness texture/gradients.
            l_region = l[region_mask > 0]
            has_texture = False
            if len(l_region) > 100:
                sample_size = min(1000, len(l_region))
                l_sample = np.random.choice(l_region, sample_size, replace=False)
                l_variance = np.var(l_sample)
                has_texture = l_variance > 300

            if is_multicolored or has_varied_saturation or has_texture:
                return True

        return False

    @staticmethod
    def _collect_traits(image):
        """Gather the numeric traits used to categorise the page."""
        traits = {}

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        h, s, v = cv2.split(hsv)
        l, a, b = cv2.split(lab)

        # Saturation profile (percentile is dtype-invariant on the uint8 channel).
        traits['mean_saturation'] = np.mean(s)
        traits['saturation_std'] = np.std(s)
        traits['high_saturation_ratio'] = np.sum(s > 50) / s.size
        traits['mid_saturation_ratio'] = np.sum((s >= 25) & (s <= 90)) / s.size
        traits['sat_percentile_90'] = np.percentile(s, 90)
        traits['sat_percentile_95'] = np.percentile(s, 95)

        # Colour spread, independent of brightness.
        traits['color_deviation_a'] = np.std(a)
        traits['color_deviation_b'] = np.std(b)
        traits['total_color_deviation'] = traits['color_deviation_a'] + traits['color_deviation_b']

        # Edge density (dense thin edges hint at text).
        edges = cv2.Canny(gray, 50, 150)
        traits['edge_density'] = np.sum(edges > 0) / edges.size

        # Lightness after a rough illumination correction.
        l_flat = l.astype(float)
        blur_size = max(l.shape) // 15
        if blur_size % 2 == 0:
            blur_size += 1
        illumination = cv2.GaussianBlur(l, (blur_size, blur_size), 0)
        l_corrected = np.clip(l_flat / (illumination.astype(float) + 1) * 128, 0, 255)
        traits['corrected_lightness'] = np.mean(l_corrected)

        # Colour clustering (white paper clusters tightly in ab-space).
        ab_flat = np.column_stack([a.flatten(), b.flatten()])
        sample_size = min(10000, ab_flat.shape[0])
        ab_sample = ab_flat[np.random.choice(ab_flat.shape[0], sample_size, replace=False)]
        traits['color_cluster_spread'] = np.std(ab_sample)

        # Share of near-zero-saturation pixels.
        hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
        low_sat_ratio = np.sum(hist_s[:30]) / np.sum(hist_s)
        traits['low_saturation_ratio'] = low_sat_ratio

        return traits

    @staticmethod
    def _decide_category(traits, has_big_color_area=False):
        """
        Turn the traits into a category label.

        - White page: low saturation, tight colour cluster, lots of text edges.
        - Colour product: high saturation / spread, or a big colour region.
        """
        # A large multicoloured region always wins.
        if has_big_color_area:
            return 'colored_product'

        is_low_saturation = traits['mean_saturation'] < 40
        is_low_color_deviation = traits['total_color_deviation'] < 20
        is_high_low_sat_ratio = traits['low_saturation_ratio'] > 0.7

        is_high_saturation = traits['mean_saturation'] > 60
        is_high_color_deviation = traits['total_color_deviation'] > 30
        has_saturated_regions = traits['high_saturation_ratio'] > 0.3

        if is_low_saturation and is_low_color_deviation and is_high_low_sat_ratio:
            return 'white_document'

        elif is_high_saturation or (is_high_color_deviation and has_saturated_regions):
            return 'colored_product'

        # Ambiguous but fairly neutral -> treat as a white page.
        if traits['total_color_deviation'] < 25:
            return 'white_document'
        else:
            return 'colored_product'

    @staticmethod
    def _suggest_threshold(traits, category):
        """Pick a per-image saturation cut-off (0-255)."""
        if category == 'white_document':
            return PageCategoryClassifier._adaptive_white_threshold(traits)

        elif category == 'colored_product':
            # Regional protection is off for colour products, so this is unused.
            return 40

        return PageCategoryClassifier._adaptive_white_threshold(traits)

    @staticmethod
    def _adaptive_white_threshold(traits):
        """
        Raise/lower the saturation cut-off for white pages so the background
        bleaches hard without wiping the occasional coloured pixel.
        """
        lightness = traits.get('corrected_lightness', 170.0)
        mid_ratio = traits.get('mid_saturation_ratio', 0.0)
        sat_p90 = traits.get('sat_percentile_90', 0.0)
        high_sat_ratio = traits.get('high_saturation_ratio', 0.0)

        base_threshold = 75.0

        # Darker paper needs harder bleaching -> raise the cut-off.
        lightness_adjust = np.clip((150.0 - lightness) / 2.5, -6.0, 22.0)

        # Lots of mid-saturation usually means tinted paper -> push higher.
        mid_ratio_adjust = np.clip(mid_ratio * 35.0, 0.0, 18.0)

        # Even the 90th percentile is low -> safe to raise more.
        tail_adjust = np.clip((60.0 - sat_p90) * 0.25, -8.0, 10.0)

        # Back off if there is real saturated content to protect.
        color_safety = -20.0 * high_sat_ratio

        threshold = base_threshold + lightness_adjust + mid_ratio_adjust + tail_adjust + color_safety
        return int(np.clip(threshold, 65, 115))


# ---------------------------------------------------------------------------
# Base single-pass page cleaner
# ---------------------------------------------------------------------------

class BasePageCleaner:
    """
    The plain clean-up pipeline (shadow removal -> denoise -> optional white
    balance) used as the "full strength" stream inside the main filter.
    """

    def __init__(self,
                 shadow_strength=1.0,
                 smoothing_level=1,
                 contrast_enhance=2.0,
                 color_correct=True):
        """
        Args:
            shadow_strength: illumination-flattening intensity (0.0-2.0).
            smoothing_level: denoise level (0-5).
            contrast_enhance: CLAHE clip limit (1.0-4.0).
            color_correct: apply grey-world white balance.
        """
        self.shadow_strength = shadow_strength
        self.smoothing_level = smoothing_level
        self.contrast_enhance = contrast_enhance
        self.color_correct = color_correct

    def clean(self, image):
        """Run the base clean-up and return (result, per-stage snapshots)."""
        stages = {}
        stages['original'] = image.copy()

        # 1. Flatten shadows.
        shadow_removed = self._flatten_illumination(image)
        stages['shadow_removed'] = shadow_removed

        # 2. Denoise.
        smoothed = self._denoise(shadow_removed)
        stages['smoothed'] = smoothed

        # 3. Optional white balance (contrast stage intentionally skipped).
        if self.color_correct:
            final = self._grey_world_balance(smoothed)
            stages['color_corrected'] = final
        else:
            final = smoothed

        stages['final'] = final

        return final, stages

    def _flatten_illumination(self, image):
        """Even out lighting via LAB L-channel illumination division."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Estimate the illumination surface from a heavy blur of L.
        blur_size = max(l.shape) // 15
        if blur_size % 2 == 0:
            blur_size += 1

        illumination = cv2.GaussianBlur(l, (blur_size, blur_size), 0)

        # Divide out the illumination (safely).
        l_corrected = guarded_ratio(
            l.astype(float),
            illumination.astype(float),
            epsilon=1.0,
            default=l.astype(float)
        )

        # Blend by shadow strength.
        l_corrected = l * (1 - self.shadow_strength) + l_corrected * 255 * self.shadow_strength
        l_corrected = np.clip(l_corrected, 0, 255).astype(np.uint8)

        lab_corrected = cv2.merge([l_corrected, a, b])
        return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

    def _denoise(self, image):
        """Denoise with strength picked by `smoothing_level`."""
        if self.smoothing_level == 0:
            return image

        if self.smoothing_level == 1:
            result = cv2.bilateralFilter(image, 5, 25, 25)
        elif self.smoothing_level == 2:
            result = cv2.bilateralFilter(image, 7, 50, 50)
        elif self.smoothing_level == 3:
            result = cv2.bilateralFilter(image, 9, 75, 75)
        elif self.smoothing_level == 4:
            result = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:  # level 5
            result = cv2.fastNlMeansDenoisingColored(image, None, 15, 15, 7, 21)

        return result

    def _boost_contrast(self, image):
        """CLAHE contrast boost on the L channel (kept for completeness)."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.contrast_enhance,
            tileGridSize=(8, 8)
        )
        l_enhanced = clahe.apply(l)

        lab_enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def _grey_world_balance(self, image):
        """Half-strength grey-world white balance in LAB."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        avg_a = np.average(lab[:, :, 1])
        avg_b = np.average(lab[:, :, 2])

        lab[:, :, 1] = lab[:, :, 1] - ((avg_a - 128) * 0.5)
        lab[:, :, 2] = lab[:, :, 2] - ((avg_b - 128) * 0.5)

        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Main auto-doc colour filter
# ---------------------------------------------------------------------------

class CrispenEngine:
    """
    Top-level Crispen engine.

    Classifies the page, then routes it through a three-zone pipeline that
    protects real colour/content while bleaching the paper background to white
    and deepening ink. No manual tuning required.
    """

    def __init__(self,
                 shadow_strength=1.0,
                 smoothing_level=1,
                 contrast_enhance=2.0,
                 color_correct=True,
                 saturation_threshold=40,
                 enable_protection=True,
                 auto_detect=True,
                 iterations=2):
        """
        Args:
            auto_detect: auto-classify the page and self-configure.
            saturation_threshold: colour-protection cut-off (0-255).
            enable_protection: protect coloured regions from bleaching.
            (other args feed the underlying BasePageCleaner)
        """
        self.auto_detect = auto_detect
        self.saturation_threshold = saturation_threshold
        self.enable_protection = enable_protection
        self.iterations = max(1, iterations)

        # Full-strength cleaner used for the "background" zone.
        self.base_cleaner = BasePageCleaner(
            shadow_strength=shadow_strength,
            smoothing_level=smoothing_level,
            contrast_enhance=contrast_enhance,
            color_correct=color_correct
        )

        # Remember the requested defaults for the unknown-category fallback.
        self.base_shadow_strength = shadow_strength
        self.base_smoothing_level = smoothing_level
        self.base_contrast_enhance = contrast_enhance
        self.base_color_correct = color_correct

    def run(self, image):
        """
        Full auto pipeline.

        Args:
            image: input BGR image.

        Returns:
            (cleaned_bgr, stage_gallery) where stage_gallery includes the
            detected category and every intermediate snapshot.
        """
        stages = {}
        stages['original'] = image.copy()

        # Classify + self-configure.
        if self.auto_detect:
            category, optimal_threshold = PageCategoryClassifier.categorize(image, self.saturation_threshold)
            stages['detected_type'] = category
            stages['suggested_threshold'] = optimal_threshold

            self.saturation_threshold = optimal_threshold
            self._configure_for_category(category)
        else:
            stages['detected_type'] = 'manual'
            stages['suggested_threshold'] = self.saturation_threshold

        category = stages.get('detected_type', 'manual')

        # Colour products just get a global auto-tone stretch.
        if category == 'colored_product' and self.auto_detect:
            stages['contrast_method'] = 'histogram_based'
            final = auto_tone_stretch(image, clip_hist_percent=10)
            stages['final'] = final
            return final, stages

        # -------------------------------------------------------------------
        # Three-zone system (priority passthrough / gentle / full clean).
        # -------------------------------------------------------------------
        priority_map_for_stages = None
        light_text_mask_for_stages = None

        if not self.enable_protection or category == 'colored_product':
            # Protection off or colour product -> simple path.
            if category == 'colored_product' and self.auto_detect:
                stages['contrast_method'] = 'histogram_based'
                final = auto_tone_stretch(image, clip_hist_percent=10)
                stages['final'] = final
                return final, stages
            else:
                final, base_stages = self.base_cleaner.clean(image)
                stages.update(base_stages)
        else:
            # Build the colour-priority map that steers protection.
            priority_map = build_color_priority_map(image, min_saturation_threshold=25)
            stages['importance_map_raw'] = (priority_map * 255).astype(np.uint8)

            # Boost graphic elements (buttons / logos / stamps / borders).
            priority_map = flag_graphic_elements(image, priority_map)
            stages['importance_map_ui'] = (priority_map * 255).astype(np.uint8)
            priority_map_for_stages = priority_map

            # Detect light text sitting on colour (whiten, don't darken).
            light_text_mask = find_light_text_on_color(image, priority_map)
            stages['white_on_color_mask'] = (light_text_mask * 255).astype(np.uint8)
            light_text_mask_for_stages = light_text_mask

            # Three streams:
            # A) untouched original (priority > 0.8)
            untouched = image.copy()

            # B) gentle clean (priority 0.3-0.8)
            gentle_cleaned = self._gentle_clean(image)
            stages['gentle_enhanced'] = gentle_cleaned

            # C) full clean (priority < 0.3)
            full_cleaned, base_stages = self.base_cleaner.clean(image)
            stages['enhanced_full'] = full_cleaned
            stages.update(base_stages)

            # Blend the three streams by priority.
            final = blend_by_priority(untouched, full_cleaned, gentle_cleaned, priority_map)
            stages['composited'] = final

        # Ink deepening, text sharpening, bleaching, legibility, colour revival
        # only apply to document-like pages.
        if category != 'colored_product':

            dark_mask = self._find_dark_content(image)
            should_preserve = False
            dark_mask_for_bleach = None

            if dark_mask is not None:
                if np.any(dark_mask):
                    should_preserve = self._dark_regions_worth_preserving(image, final, dark_mask)

                    if should_preserve:
                        stages['dark_mask'] = dark_mask
                        final = self._preserve_dark_regions(image, final, dark_mask)
                        stages['dark_preserved'] = final
                        dark_mask_for_bleach = dark_mask
                    else:
                        # Rejected: likely shadow, not content. Keep only the
                        # genuinely dark (<120) pixels for bleach protection.
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
                        truly_dark = (gray < 120).astype(np.uint8) * 255
                        dark_mask_filtered = cv2.bitwise_and(dark_mask, truly_dark)
                        filtered_ratio = np.count_nonzero(dark_mask_filtered) / dark_mask_filtered.size

                        if filtered_ratio >= 0.005:
                            dark_mask_for_bleach = dark_mask_filtered
                        else:
                            dark_mask_for_bleach = None

            stages['pre_darkening'] = final

            # Deepen ink strokes.
            if priority_map_for_stages is not None:
                final = deepen_ink_strokes(
                    final,
                    saturation_threshold=self.saturation_threshold,
                    color_importance_map=priority_map_for_stages,
                    dark_mask=dark_mask
                )
            else:
                final = deepen_ink_strokes(
                    final,
                    saturation_threshold=self.saturation_threshold,
                    dark_mask=dark_mask
                )
            stages['intelligent_darkening'] = final

            # Sharpen written content.
            if priority_map_for_stages is not None:
                final = sharpen_written_content(
                    final,
                    saturation_threshold=self.saturation_threshold,
                    color_importance_map=priority_map_for_stages,
                    white_on_color_mask=light_text_mask_for_stages,
                    dark_mask=dark_mask
                )
            else:
                final = sharpen_written_content(
                    final,
                    saturation_threshold=self.saturation_threshold,
                    dark_mask=dark_mask
                )
            stages['text_enhanced'] = final

            # Build a content-protection mask for the bleaching pass.
            content_protection_mask = None
            if priority_map_for_stages is not None:
                gray_for_protection = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY).astype(np.float32)
                norm_gray_protection = gray_for_protection / 255.0

                high_importance = (priority_map_for_stages > 0.3).astype(np.float32)

                edges = cv2.Canny(gray_for_protection.astype(np.uint8), 40, 120)
                edges_dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
                has_edges = (edges_dilated > 0).astype(np.float32)

                content_protection_mask = np.maximum(high_importance, has_edges)
                try:
                    content_protection_mask = _boundary_preserving_blur(content_protection_mask, norm_gray_protection, radius_ratio=0.02, eps=1e-3)
                except Exception:
                    kernel_size = _scale_blur_window(final.shape, base_ratio=0.02, min_size=9)
                    content_protection_mask = cv2.GaussianBlur(content_protection_mask, (kernel_size, kernel_size), 0)
                content_protection_mask = np.clip(content_protection_mask, 0.0, 1.0)

            # Bleach the paper background.
            bleach_result = bleach_paper_background(
                final,
                whitening_strength=0.5,
                color_importance_map=priority_map_for_stages,
                dark_mask=dark_mask_for_bleach,
                content_protection_mask=content_protection_mask,
                collect_debug=True
            )
            if isinstance(bleach_result, tuple):
                final, bleach_debug = bleach_result
            else:
                final = bleach_result
                bleach_debug = None
            if bleach_debug:
                stages.update(bleach_debug)
            stages['background_whitened'] = final

            # Boost content legibility against the new white background.
            if priority_map_for_stages is not None:
                final = boost_content_legibility(
                    final,
                    color_importance_map=priority_map_for_stages,
                    dark_mask=dark_mask_for_bleach if dark_mask_for_bleach is not None else None,
                    content_protection_mask=content_protection_mask
                )
            else:
                final = boost_content_legibility(
                    final,
                    dark_mask=dark_mask_for_bleach if dark_mask_for_bleach is not None else None
                )
            stages['content_visibility_enhanced'] = final

            # Revive the page's original colours.
            if priority_map_for_stages is not None:
                final = revive_original_colors(
                    final,
                    color_importance_map=priority_map_for_stages,
                    saturation_boost=1.5,
                    vibrancy_boost=1.3
                )
            else:
                final = revive_original_colors(final, saturation_boost=1.5, vibrancy_boost=1.3)
            stages['colors_boosted'] = final

        # Stop at colour revival - defect repair / iterative passes are skipped
        # deliberately to keep the output crisp.
        stages['iterations'] = 1
        stages['final'] = final
        return final, stages

    def _configure_for_category(self, category):
        """Pick internal parameters for the detected page category."""
        if category == 'white_document':
            # Aggressive clean + white balance, protection on for stray colour.
            self.base_cleaner.shadow_strength = 1.0
            self.base_cleaner.smoothing_level = 1
            self.base_cleaner.contrast_enhance = 2.0
            self.base_cleaner.color_correct = True
            self.enable_protection = True

        elif category == 'colored_product':
            # Gentle, no white balance, no regional protection.
            self.base_cleaner.shadow_strength = 0.4
            self.base_cleaner.smoothing_level = 1
            self.base_cleaner.contrast_enhance = 1.5
            self.base_cleaner.color_correct = False
            self.enable_protection = False

        else:
            # Unknown -> restore the caller's defaults.
            self.base_cleaner.shadow_strength = self.base_shadow_strength
            self.base_cleaner.smoothing_level = self.base_smoothing_level
            self.base_cleaner.contrast_enhance = self.base_contrast_enhance
            self.base_cleaner.color_correct = self.base_color_correct
            self.enable_protection = False

    def _find_color_regions(self, image):
        """Binary mask of saturated regions (255 = colour, 0 = neutral)."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        color_mask = (s > self.saturation_threshold).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

        gray_for_guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        try:
            color_mask_float = color_mask.astype(np.float32) / 255.0
            color_mask_float = _boundary_preserving_blur(color_mask_float, gray_for_guide, radius_ratio=0.02, eps=1e-3)
            color_mask = (color_mask_float * 255.0).astype(np.uint8)
        except Exception:
            color_mask = cv2.GaussianBlur(color_mask, (3, 3), 0)

        return color_mask

    def _gentle_clean(self, image):
        """Very light clean for the medium-priority zone - colours kept intact."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        blur_size = max(l.shape) // 20  # wider blur = gentler
        if blur_size % 2 == 0:
            blur_size += 1

        illumination = cv2.GaussianBlur(l, (blur_size, blur_size), 0)
        l_corrected = guarded_ratio(
            l.astype(float),
            illumination.astype(float),
            epsilon=1.0,
            default=l.astype(float)
        )

        gentle_strength = 0.1
        l_corrected = l * (1 - gentle_strength) + l_corrected * 255 * gentle_strength
        l_corrected = np.clip(l_corrected, 0, 255).astype(np.uint8)

        lab_corrected = cv2.merge([l_corrected, a, b])
        shadow_removed = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

        # Minimal denoise, no contrast/white-balance to preserve colour.
        smoothed = cv2.bilateralFilter(shadow_removed, 3, 15, 15)

        return smoothed

    def _find_dark_content(self, image):
        """
        Locate large, genuinely dark regions (portraits, illustrations). Thin
        text strokes are filtered out so legibility work isn't disturbed.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 7)
        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            thresh, connectivity=8
        )
        min_area = int(0.005 * image.shape[0] * image.shape[1])
        dark_mask = np.zeros_like(thresh, dtype=np.uint8)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= min_area:
                dark_mask[labels == label] = 255

        if not np.any(dark_mask):
            return None

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel_close)

        gray_for_guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        dark_mask_float = dark_mask.astype(np.float32) / 255.0
        try:
            dark_mask_float = _boundary_preserving_blur(dark_mask_float, gray_for_guide, radius_ratio=0.02, eps=1e-3)
            dark_mask = (dark_mask_float * 255.0).astype(np.uint8)
        except Exception:
            dark_mask = cv2.GaussianBlur(dark_mask, (3, 3), 0)
        return dark_mask

    def _preserve_dark_regions(self, original, cleaned, mask):
        """
        Blend the original back into dark regions (logos/photos) so they don't
        turn muddy - darker areas keep more of the original.
        """
        mask_norm = (mask.astype(np.float32) / 255.0)[..., None]

        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)

        orig_brightness = orig_gray / 255.0
        # Very dark regions keep 80-100% original; mid-dark blend more.
        adaptive_blend = np.clip(0.5 + (1.0 - orig_brightness) * 0.5, 0.5, 1.0)
        adaptive_blend = adaptive_blend[..., None]

        effective_mask = mask_norm * adaptive_blend
        preserved = (
            original.astype(np.float32) * effective_mask +
            cleaned.astype(np.float32) * (1.0 - effective_mask)
        )
        return np.clip(preserved, 0, 255).astype(np.uint8)

    def _dark_regions_worth_preserving(self, original, cleaned, mask):
        """
        Decide whether dark-region preservation should run - avoids harming
        clean pages where the detection was just noise.
        """
        mask_pixels = mask > 0
        mask_ratio = np.count_nonzero(mask_pixels) / mask.size
        if mask_ratio < 0.01:
            return False  # negligible, likely text noise

        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
        out_gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY).astype(np.float32)

        orig_mean = np.mean(orig_gray[mask_pixels])
        out_mean = np.mean(out_gray[mask_pixels])
        brightening = out_mean - orig_mean

        # Require genuinely dark content that got noticeably brightened.
        return orig_mean < 120 and brightening > 15


# ---------------------------------------------------------------------------
# Global tone / colour-space converters
# ---------------------------------------------------------------------------

def auto_tone_stretch(src, clip_hist_percent=10):
    """
    Photoshop-style "auto contrast": stretch intensities to fill 0-255 after
    clipping a percentage off each histogram tail.

    Args:
        src: input BGR (8UC1/8UC3/8UC4).
        clip_hist_percent: total percent to clip from both tails.
    """
    hist_size = 256
    alpha = 1.0
    beta = 0.0
    min_gray = 0
    max_gray = 0

    if len(src.shape) == 2 or src.shape[2] == 1:
        gray = src.copy()
    elif src.shape[2] == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    elif src.shape[2] == 4:
        gray = cv2.cvtColor(src, cv2.COLOR_BGRA2GRAY)
    else:
        return src

    if clip_hist_percent == 0:
        min_gray, max_gray = np.min(gray), np.max(gray)
    else:
        hist = cv2.calcHist([gray], [0], None, [hist_size], [0, 256])

        accumulator = np.zeros(hist_size, dtype=np.float32)
        accumulator[0] = hist[0, 0]
        for i in range(1, hist_size):
            accumulator[i] = accumulator[i - 1] + hist[i, 0]

        max_val = accumulator[-1]
        clip_abs = clip_hist_percent * (max_val / 100.0)
        clip_abs /= 2.0  # split across both tails

        min_gray = 0
        while accumulator[min_gray] < clip_abs:
            min_gray += 1

        max_gray = hist_size - 1
        while accumulator[max_gray] >= (max_val - clip_abs):
            max_gray -= 1

    input_range = max_gray - min_gray
    if input_range > 0:
        alpha = (hist_size - 1) / input_range
        beta = -min_gray * alpha
    else:
        alpha = 1.0
        beta = 0.0

    dst = src.copy().astype(np.float32)
    dst = alpha * dst + beta
    dst = np.clip(dst, 0, 255).astype(np.uint8)

    return dst


def pil_to_bgr(pil_image):
    """Convert a PIL image to an OpenCV BGR array."""
    from PIL import Image  # local import keeps the core free of hard PIL dep
    img_array = np.array(pil_image)
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return img_array


def bgr_to_pil(cv2_image):
    """Convert an OpenCV BGR array to a PIL image."""
    from PIL import Image
    if len(cv2_image.shape) == 3:
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_image)


# ---------------------------------------------------------------------------
# Vignette / shadow detection
# ---------------------------------------------------------------------------

def find_edge_vignette(image, gray, norm_gray):
    """
    Find dark border vignettes (lighting fall-off at page edges) that must NOT
    be treated as content. Returns a float mask where 1.0 = vignette.

    A vignette is: near a border, smoothly dark, low saturation, gradual, and
    connected to the border rather than enclosed.
    """
    height, width = gray.shape

    # Border zone.
    border_thickness = max(min(height, width) // 8, 30)
    border_mask = np.zeros_like(norm_gray, dtype=np.float32)
    border_mask[0:border_thickness, :] = 1.0
    border_mask[height-border_thickness:height, :] = 1.0
    border_mask[:, 0:border_thickness] = 1.0
    border_mask[:, width-border_thickness:width] = 1.0

    kernel_size = border_thickness * 2 + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    border_mask = cv2.GaussianBlur(border_mask, (kernel_size, kernel_size), 0)

    # Criterion 1: mid-dark band (adaptive to brightness).
    # np.percentile flattens internally, so skip the explicit copy.
    brightness_p25 = np.percentile(norm_gray, 25)
    brightness_p50 = np.percentile(norm_gray, 50)
    shadow_lower = np.clip(brightness_p25 * 0.8, 0.15, 0.40)
    shadow_upper = np.clip(brightness_p50 * 1.1, 0.50, 0.75)
    is_dark = ((norm_gray > shadow_lower) & (norm_gray < shadow_upper)).astype(np.float32)

    # Criterion 2: low saturation.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    norm_s = s.astype(np.float32) / 255.0
    sat_p75 = np.percentile(norm_s, 75)
    adaptive_sat_threshold = np.clip(sat_p75 * 1.2, 0.15, 0.30)
    is_low_sat = (norm_s < adaptive_sat_threshold).astype(np.float32)

    # Criterion 3: smooth (low texture).
    kernel = np.ones((21, 21), np.float32) / 441
    mean_local = cv2.filter2D(norm_gray, -1, kernel)
    variance_local = cv2.filter2D((norm_gray - mean_local) ** 2, -1, kernel)
    variance_p50 = np.percentile(variance_local, 50)
    adaptive_variance_threshold = np.clip(variance_p50 * 1.5, 0.01, 0.03)
    is_smooth = (variance_local < adaptive_variance_threshold).astype(np.float32)

    # Criterion 4: gradual gradient (not sharp content edges).
    gradient_kernel = np.array([[-1, 0, 1]], dtype=np.float32)
    grad_x = cv2.filter2D(norm_gray, -1, gradient_kernel)
    grad_y = cv2.filter2D(norm_gray, -1, gradient_kernel.T)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_p50 = np.percentile(gradient_magnitude, 50)
    grad_p75 = np.percentile(gradient_magnitude, 75)
    grad_low = np.clip(grad_p50 * 0.7, 0.02, 0.06)
    grad_high = np.clip(grad_p75 * 1.3, 0.15, 0.30)
    is_smooth_gradient = ((gradient_magnitude > grad_low) & (gradient_magnitude < grad_high)).astype(np.float32)

    # Criterion 5: connected to a border.
    dark_candidate = (is_dark * is_low_sat * is_smooth * is_smooth_gradient).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_candidate, connectivity=8)

    border_connected_mask = np.zeros_like(norm_gray, dtype=np.float32)
    min_area = int(height * width * 0.001)

    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] < min_area:
            continue

        label_mask = (labels == label).astype(np.uint8)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_thickness // 2, border_thickness // 2))
        label_dilated = cv2.dilate(label_mask, kernel_dilate, iterations=1)

        border_overlap = np.sum((label_dilated > 0) & (border_mask > 0.3))
        total_label = np.sum(label_mask > 0)

        if border_overlap > 0 and total_label > 0:
            overlap_ratio = border_overlap / total_label
            if overlap_ratio > 0.3:
                border_connected_mask[label_mask > 0] = 1.0

    # Combine every criterion.
    vignette_mask = (
        border_mask *
        is_dark *
        is_low_sat *
        is_smooth *
        is_smooth_gradient *
        border_connected_mask
    )

    # Knock out high edge-density areas (text/graphics, not vignette).
    edges = cv2.Canny(gray, 40, 120)
    edge_density = cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (15, 15), 0)
    high_edge_regions = (edge_density > 0.12).astype(np.float32)
    vignette_mask = vignette_mask * (1.0 - high_edge_regions * 0.9)

    kernel_size = max(31, min(height, width) // 20)
    if kernel_size % 2 == 0:
        kernel_size += 1
    vignette_mask = cv2.GaussianBlur(vignette_mask, (kernel_size, kernel_size), 0)

    return np.clip(vignette_mask, 0.0, 1.0)


def _tighten_region_mask(mask, guide, erosion_radius=3, edge_stop=0.85, radius_ratio=0.008):
    """
    Shrink a soft mask so it hugs content boundaries tightly (kills grey halos).

    - light erosion trims padded margins;
    - an edge penalty halts the mask at strong gradients;
    - guided smoothing keeps fills soft without bleeding over edges.
    """
    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)

    if erosion_radius > 0:
        kernel_size = erosion_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.erode(mask, kernel, iterations=1)

    guide_u8 = np.clip(guide * 255.0, 0, 255).astype(np.uint8)
    edges = cv2.Canny(guide_u8, 60, 160)
    try:
        edge_penalty = _boundary_preserving_blur(edges.astype(np.float32) / 255.0, guide, radius_ratio=0.01, eps=1e-3)
    except Exception:
        edge_penalty = cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (3, 3), 0)
    mask = mask * (1.0 - np.clip(edge_penalty * edge_stop, 0.0, 1.0))

    try:
        mask = _boundary_preserving_blur(mask, guide, radius_ratio=radius_ratio, eps=1e-3)
    except Exception:
        kernel_size = max(3, erosion_radius * 2 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        mask = cv2.GaussianBlur(mask, (min(5, kernel_size), min(5, kernel_size)), 0)

    return np.clip(mask, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Colour-priority map
# ---------------------------------------------------------------------------

def build_color_priority_map(image, min_saturation_threshold=25):
    """
    Build a per-pixel colour-priority map (0.0-1.0) that tells the cleaner what
    to protect:

    - 1.0: critical colour -> complete passthrough
    - 0.5-0.8: moderate -> gentle clean
    - 0.0-0.5: low -> full clean

    It combines several triggers (any of which raises priority): high
    saturation, distinct hue, uniform colour blobs, coloured edges, dark
    saturated colours - then subtracts border vignettes and smooth dark
    shadows, and finally tightens the mask to real content.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)

    norm_gray = gray.astype(np.float32) / 255.0

    # Border vignettes get their priority stripped later.
    vignette_mask = find_edge_vignette(image, gray, norm_gray)

    # Distance-to-border penalty (lower priority near edges).
    height, width = gray.shape
    border_thickness = max(min(height, width) // 6, 40)

    y_coords = np.arange(height, dtype=np.float32)[:, None]
    x_coords = np.arange(width, dtype=np.float32)[None, :]

    dist_top = y_coords / border_thickness
    dist_bottom = (height - y_coords) / border_thickness
    dist_left = x_coords / border_thickness
    dist_right = (width - x_coords) / border_thickness

    dist_to_border = np.minimum(
        np.minimum(dist_top, dist_bottom),
        np.minimum(dist_left, dist_right)
    )

    border_penalty = np.clip(dist_to_border, 0.0, 1.0)
    border_penalty = 0.3 + border_penalty * 0.7  # 0.3 at edge, 1.0 at centre

    # Multi-scale texture (dark+textured = content, dark+smooth = shadow).
    kernel_small = np.ones((9, 9), np.float32) / 81
    kernel_large = np.ones((21, 21), np.float32) / 441

    mean_local_small = cv2.filter2D(norm_gray, -1, kernel_small)
    variance_local_small = cv2.filter2D((norm_gray - mean_local_small) ** 2, -1, kernel_small)

    mean_local_large = cv2.filter2D(norm_gray, -1, kernel_large)
    variance_local_large = cv2.filter2D((norm_gray - mean_local_large) ** 2, -1, kernel_large)

    variance_p50 = np.percentile(variance_local_large, 50)
    texture_threshold = np.clip(variance_p50 * 1.2, 0.01, 0.025)

    is_dark = (norm_gray < 0.6).astype(np.float32)
    has_texture = (variance_local_large > texture_threshold).astype(np.float32)
    dark_without_texture = is_dark * (1.0 - has_texture)

    edges = cv2.Canny(gray, 30, 120)
    edges_binary = cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (5, 5), 0)

    priority = np.zeros(gray.shape, dtype=np.float32)

    # --- Trigger 1: high saturation --------------------------------------
    # np.percentile is dtype/shape-invariant for these values, so read it
    # straight off the uint8 saturation channel (no flatten/astype copy).
    sat_p75 = np.percentile(s, 75)
    adaptive_sat_threshold = max(min_saturation_threshold, min(sat_p75, 40))

    sat_norm = s.astype(np.float32) / 255.0
    high_sat_mask = _soft_ramp(
        adaptive_sat_threshold / 255.0,
        min(1.0, (adaptive_sat_threshold + 25) / 255.0),
        sat_norm
    )
    priority = np.maximum(priority, high_sat_mask * 0.7)

    very_high_sat = _soft_ramp(
        max(adaptive_sat_threshold + 15, 55) / 255.0,
        min(1.0, (adaptive_sat_threshold + 50) / 255.0),
        sat_norm
    )
    priority = np.maximum(priority, very_high_sat)

    # --- Trigger 2: distinct hue -----------------------------------------
    h_float = h.astype(np.float32)
    hue_strength = np.minimum(h_float / 90.0, (180.0 - h_float) / 90.0)
    hue_strength = 1.0 - hue_strength

    colorfulness = hue_strength * sat_norm * (v.astype(np.float32) / 255.0)
    distinct_hue_mask = _soft_ramp(0.2, 0.45, colorfulness)

    # Exclude bright, low-saturation paper (cream/beige) from hue triggering.
    is_bright = (v.astype(np.float32) / 255.0 > 0.7).astype(np.float32)
    has_low_sat = (sat_norm < 0.15).astype(np.float32)
    is_background = is_bright * has_low_sat
    distinct_hue_mask = distinct_hue_mask * (1.0 - is_background * 0.95)

    priority = np.maximum(priority, distinct_hue_mask * 0.6)

    # --- Trigger 3: uniform colour blobs (buttons/logos) -----------------
    kernel_size = _scale_blur_window(image.shape, base_ratio=0.015, min_size=9)
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

    a_mean = cv2.filter2D(a.astype(np.float32), -1, kernel)
    b_mean = cv2.filter2D(b.astype(np.float32), -1, kernel)

    a_var = cv2.filter2D((a.astype(np.float32) - a_mean) ** 2, -1, kernel)
    b_var = cv2.filter2D((b.astype(np.float32) - b_mean) ** 2, -1, kernel)
    variance_sum = np.maximum(a_var + b_var, 0.0)
    local_variance = np.sqrt(variance_sum)
    local_variance = scrub_invalid_values(local_variance, fill_value=0.0)

    a_dist = np.abs(a_mean - 128.0)
    b_dist = np.abs(b_mean - 128.0)
    color_distance_sq = np.maximum(a_dist ** 2 + b_dist ** 2, 0.0)
    color_distance = np.sqrt(color_distance_sq)
    color_distance = scrub_invalid_values(color_distance, fill_value=0.0)

    # Exclude tinted paper: needs real saturation, not just a colour cast.
    local_sat_mean = cv2.filter2D(s.astype(np.float32), -1, kernel)
    has_real_saturation = (local_sat_mean > 35).astype(np.float32)

    local_l_mean = cv2.filter2D(l.astype(np.float32), -1, kernel)
    is_bright_background = (local_l_mean > 180).astype(np.float32)
    is_not_background = 1.0 - is_bright_background * 0.9

    has_both_channels = (a_dist > 8).astype(np.float32)

    variance_score = _soft_ramp(4, 18, np.maximum(0, 18 - local_variance))
    distance_score = _soft_ramp(15, 35, color_distance)
    uniform_color_mask = variance_score * distance_score

    # Require some texture so smooth shadows don't read as colour blobs.
    has_texture_detail = (variance_local_small > texture_threshold * 0.5).astype(np.float32)
    texture_requirement = np.maximum(has_texture_detail, 0.3)

    uniform_color_mask = uniform_color_mask * has_real_saturation * is_not_background * has_both_channels * texture_requirement
    priority = np.maximum(priority, uniform_color_mask * 0.85)

    # --- Trigger 4: coloured edges (UI elements) -------------------------
    gradient_kernel = np.array([[-1, 0, 1]], dtype=np.float32)
    grad_x = cv2.filter2D(norm_gray, -1, gradient_kernel)
    grad_y = cv2.filter2D(norm_gray, -1, gradient_kernel.T)

    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2) + 1e-6
    grad_x_norm = grad_x / grad_magnitude
    grad_y_norm = grad_y / grad_magnitude

    height, width = gray.shape
    y_coords = np.arange(height, dtype=np.float32)[:, None]
    x_coords = np.arange(width, dtype=np.float32)[None, :]

    dist_to_top = y_coords
    dist_to_bottom = height - y_coords
    dist_to_left = x_coords
    dist_to_right = width - x_coords

    border_dir_y = np.where(dist_to_top < dist_to_bottom, -1.0, 1.0)
    border_dir_x = np.where(dist_to_left < dist_to_right, -1.0, 1.0)

    # Edges pointing away from the border are content, not shadow boundaries.
    edge_away_from_border = 1.0 - np.abs(grad_x_norm * border_dir_x + grad_y_norm * border_dir_y) * 0.5
    edge_away_from_border = np.clip(edge_away_from_border, 0.3, 1.0)

    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_dilated = cv2.dilate(edges_binary, kernel_erode, iterations=1)

    has_color = _soft_ramp(min_saturation_threshold / 255.0, (min_saturation_threshold + 30) / 255.0, sat_norm)
    edge_color_mask = edges_dilated * has_color * edge_away_from_border
    priority = np.maximum(priority, edge_color_mask * 0.8)

    # --- Trigger 5: dark saturated colours (stamps, dark logos) ----------
    brightness_norm = v.astype(np.float32) / 255.0
    dark_saturated = _soft_ramp(70 / 255.0, 110 / 255.0, sat_norm) * _soft_ramp(0.4, 0.7, 1.0 - brightness_norm)
    priority = np.maximum(priority, dark_saturated * 0.9)

    # --- Smooth + normalise ----------------------------------------------
    try:
        priority = _boundary_preserving_blur(priority, gray, radius_ratio=0.04, eps=1e-3)
    except Exception:
        kernel_size = _scale_blur_window(image.shape, base_ratio=0.03, min_size=9)
        priority = cv2.GaussianBlur(priority, (kernel_size, kernel_size), 0)
    priority = np.clip(priority, 0.0, 1.0)

    # Strip border vignettes.
    priority = priority * (1.0 - vignette_mask * 0.99)

    # Reduce priority on smooth dark regions (shadows).
    dark_shadow_penalty = dark_without_texture * 0.7
    priority = priority * (1.0 - dark_shadow_penalty)

    # Apply border penalty, but keep saturated+sharp content near edges.
    high_sat_near_border = (sat_norm > 0.3).astype(np.float32)
    sharp_edges_near_border = (edges_binary > 0.5).astype(np.float32)
    important_near_border = high_sat_near_border * sharp_edges_near_border

    border_penalty_applied = border_penalty * (1.0 - important_near_border * 0.5) + important_near_border * 1.0
    border_penalty_applied = np.clip(border_penalty_applied, 0.3, 1.0)
    priority = priority * border_penalty_applied

    # Final paper-background exclusion (bright + low sat should bleach).
    brightness_norm = v.astype(np.float32) / 255.0
    is_very_bright = (brightness_norm > 0.75).astype(np.float32)
    has_very_low_sat = (sat_norm < 0.12).astype(np.float32)
    is_paper_background = is_very_bright * has_very_low_sat

    l_norm = l.astype(np.float32) / 255.0
    is_very_bright_lab = (l_norm > 0.75).astype(np.float32)
    is_paper_background = is_paper_background * is_very_bright_lab

    background_exclusion = 1.0 - is_paper_background * 0.98
    priority = priority * background_exclusion
    priority = np.clip(priority, 0.0, 1.0)

    # Sharpen + tighten to shrink margins.
    priority = np.power(priority, 1.15, where=(priority > 0), out=np.zeros_like(priority))
    priority = _tighten_region_mask(
        priority,
        norm_gray,
        erosion_radius=3,
        edge_stop=0.85,
        radius_ratio=0.008
    )

    # Guard against over-protection using precise content detection.
    content_guard = _locate_all_content(
        image,
        gray.astype(np.float32),
        norm_gray,
        priority
    )
    content_guard = _tighten_region_mask(
        content_guard,
        norm_gray,
        erosion_radius=3,
        edge_stop=0.85,
        radius_ratio=0.008
    )
    priority = priority * np.clip(content_guard, 0.0, 1.0)
    priority = np.clip(priority, 0.0, 1.0)

    return priority


def _derive_local_paper_targets(image, importance_map=None, window_ratio=0.18):
    """
    Analyse paper tone across overlapping tiles and produce smoothly varying
    per-pixel maps (target white, whitening strength, background threshold)
    that drive the bleach pass.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height, width = gray.shape

    window_size = int(max(128, min(height, width) * window_ratio))
    stride = max(window_size // 3, 64)  # more overlap = fewer seams
    tiles = _tile_scan_regions(gray.shape, window_size, stride)

    target_map = np.zeros_like(gray)
    strength_map = np.zeros_like(gray)
    background_thresh_map = np.zeros_like(gray)
    weight_map = np.zeros_like(gray)

    gaussian_window = cv2.getGaussianKernel(window_size, window_size / 4)
    gaussian_2d = gaussian_window @ gaussian_window.T

    for (y0, y1, x0, x1) in tiles:
        patch = l_channel[y0:y1, x0:x1]
        if patch.size == 0:
            continue

        patch_h, patch_w = patch.shape
        if patch_h < 10 or patch_w < 10:
            continue

        l_mean = np.mean(patch)

        # Always aim for pure white.
        target_whiteness = 255.0

        if l_mean < 140:
            background_threshold = 0.48
        elif l_mean < 170:
            background_threshold = 0.54
        elif l_mean < 200:
            background_threshold = 0.6
        else:
            background_threshold = 0.65

        distance = np.clip(target_whiteness - l_mean, 10, 255)
        local_strength = np.clip(distance / 100.0, 0.5, 3.0)

        if importance_map is not None:
            importance_patch = importance_map[y0:y1, x0:x1]
            color_penalty = np.mean(importance_patch)
            local_strength *= (1.0 - color_penalty * 0.85)

        weight_mask = gaussian_2d[:patch_h, :patch_w] if (patch_h <= window_size and patch_w <= window_size) else np.ones((patch_h, patch_w), dtype=np.float32)

        target_map[y0:y1, x0:x1] += target_whiteness * weight_mask
        strength_map[y0:y1, x0:x1] += local_strength * weight_mask
        background_thresh_map[y0:y1, x0:x1] += background_threshold * weight_mask
        weight_map[y0:y1, x0:x1] += weight_mask

    weight_map = np.maximum(weight_map, 1e-3)
    target_map = guarded_ratio(target_map, weight_map, epsilon=1e-3, default=245.0)
    strength_map = guarded_ratio(strength_map, weight_map, epsilon=1e-3, default=0.8)
    background_thresh_map = guarded_ratio(background_thresh_map, weight_map, epsilon=1e-3, default=0.6)

    target_map = scrub_invalid_values(target_map, fill_value=245.0)
    strength_map = scrub_invalid_values(strength_map, fill_value=0.8)
    background_thresh_map = scrub_invalid_values(background_thresh_map, fill_value=0.6)

    kernel_size = _scale_blur_window(image.shape, base_ratio=0.12, min_size=81)
    target_map = cv2.GaussianBlur(target_map, (kernel_size, kernel_size), 0)
    strength_map = cv2.GaussianBlur(strength_map, (kernel_size, kernel_size), 0)
    background_thresh_map = cv2.GaussianBlur(background_thresh_map, (kernel_size, kernel_size), 0)

    guide = gray / 255.0
    try:
        target_map = _boundary_preserving_blur(target_map, guide, radius_ratio=0.05, eps=1e-2)
        strength_map = _boundary_preserving_blur(strength_map, guide, radius_ratio=0.05, eps=1e-3)
        background_thresh_map = _boundary_preserving_blur(background_thresh_map, guide, radius_ratio=0.05, eps=1e-3)
    except Exception:
        target_map = cv2.GaussianBlur(target_map, (kernel_size, kernel_size), 0)
        strength_map = cv2.GaussianBlur(strength_map, (kernel_size, kernel_size), 0)
        background_thresh_map = cv2.GaussianBlur(background_thresh_map, (kernel_size, kernel_size), 0)

    target_map = scrub_invalid_values(target_map, fill_value=245.0)
    strength_map = scrub_invalid_values(strength_map, fill_value=0.8)
    background_thresh_map = scrub_invalid_values(background_thresh_map, fill_value=0.6)

    return {
        'target_map': target_map,
        'strength_map': np.clip(strength_map, 0.3, 3.0),
        'background_thresh_map': np.clip(background_thresh_map, 0.4, 0.75)
    }


def flag_graphic_elements(image, color_importance_map):
    """
    Detect buttons, logos, borders and stamps by shape/position and raise their
    priority to critical (1.0). Returns the updated priority map.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, s, v = cv2.split(hsv)

    height, width = image.shape[:2]
    enhanced_importance = color_importance_map.copy()

    # --- Buttons: rectangular colour blobs with inner text ----------------
    colored_regions = (s > 30).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    colored_regions = cv2.morphologyEx(colored_regions, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(colored_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        min_area = (width * height) * 0.001
        max_area = (width * height) * 0.15

        if min_area < area < max_area:
            rect = cv2.boundingRect(contour)
            x, y, w, rect_h = rect  # rect_h avoids clobbering the hue channel
            rect_area = w * rect_h
            extent = guarded_ratio(area, rect_area, epsilon=1e-6, default=0.0)

            aspect_ratio = guarded_ratio(max(w, rect_h), min(w, rect_h), epsilon=1e-6, default=1.0)

            if extent > 0.6 and aspect_ratio < 5.0:
                button_mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(button_mask, [contour], -1, 255, -1)

                button_region = cv2.bitwise_and(gray, button_mask)
                button_std = np.std(button_region[button_mask > 0]) if np.any(button_mask > 0) else 0
                if button_std > 30:  # contrast inside -> likely a labelled button
                    button_mask_float = (button_mask / 255.0).astype(np.float32)
                    enhanced_importance = np.maximum(enhanced_importance, button_mask_float * 1.0)

    # --- Logos: coloured corners -----------------------------------------
    corner_size = min(width, height) // 4
    top_left_region = color_importance_map[0:corner_size, 0:corner_size]
    top_right_region = color_importance_map[0:corner_size, width-corner_size:width]

    if np.mean(top_left_region) > 0.5:
        enhanced_importance[0:corner_size, 0:corner_size] = np.maximum(
            enhanced_importance[0:corner_size, 0:corner_size],
            top_left_region * 0.3 + 0.7
        )

    if np.mean(top_right_region) > 0.5:
        enhanced_importance[0:corner_size, width-corner_size:width] = np.maximum(
            enhanced_importance[0:corner_size, width-corner_size:width],
            top_right_region * 0.3 + 0.7
        )

    # --- Borders: coloured strips along edges ----------------------------
    edge_thickness = min(width, height) // 20
    top_border = color_importance_map[0:edge_thickness, :]
    if np.mean(top_border) > 0.4:
        enhanced_importance[0:edge_thickness, :] = np.maximum(
            enhanced_importance[0:edge_thickness, :], 0.9
        )

    bottom_border = color_importance_map[height-edge_thickness:height, :]
    if np.mean(bottom_border) > 0.4:
        enhanced_importance[height-edge_thickness:height, :] = np.maximum(
            enhanced_importance[height-edge_thickness:height, :], 0.9
        )

    # --- Stamps: small red/blue/purple blobs -----------------------------
    red_hue_mask = ((h < 10) | (h > 170)).astype(np.uint8)
    red_sat_mask = (s > 100).astype(np.uint8)
    red_mask = (red_hue_mask * red_sat_mask * 255).astype(np.uint8)

    blue_hue_mask = ((h > 100) & (h < 130)).astype(np.uint8)
    blue_sat_mask = (s > 80).astype(np.uint8)
    blue_mask = (blue_hue_mask * blue_sat_mask * 255).astype(np.uint8)

    purple_hue_mask = ((h > 130) & (h < 160)).astype(np.uint8)
    purple_sat_mask = (s > 80).astype(np.uint8)
    purple_mask = (purple_hue_mask * purple_sat_mask * 255).astype(np.uint8)

    stamp_mask = np.maximum(np.maximum(red_mask, blue_mask), purple_mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(stamp_mask, connectivity=8)
    min_stamp_area = (width * height) * 0.0005
    max_stamp_area = (width * height) * 0.05

    for label in range(1, num_labels):
        if min_stamp_area < stats[label, cv2.CC_STAT_AREA] < max_stamp_area:
            stamp_region = (labels == label).astype(np.float32)
            enhanced_importance = np.maximum(enhanced_importance, stamp_region * 1.0)

    try:
        enhanced_importance = _boundary_preserving_blur(enhanced_importance, gray / 255.0, radius_ratio=0.04, eps=1e-3)
    except Exception:
        kernel_size = _scale_blur_window(image.shape, base_ratio=0.03, min_size=9)
        enhanced_importance = cv2.GaussianBlur(enhanced_importance, (kernel_size, kernel_size), 0)
    enhanced_importance = scrub_invalid_values(enhanced_importance, fill_value=0.0)
    enhanced_importance = np.clip(enhanced_importance, 0.0, 1.0)

    enhanced_importance = np.power(enhanced_importance, 1.1, where=(enhanced_importance > 0), out=np.zeros_like(enhanced_importance))
    enhanced_importance = _tighten_region_mask(
        enhanced_importance,
        gray.astype(np.float32) / 255.0,
        erosion_radius=3,
        edge_stop=0.85,
        radius_ratio=0.008
    )

    content_guard = _locate_all_content(
        image,
        gray.astype(np.float32),
        gray.astype(np.float32) / 255.0,
        enhanced_importance
    )
    content_guard = _tighten_region_mask(
        content_guard,
        gray.astype(np.float32) / 255.0,
        erosion_radius=3,
        edge_stop=0.85,
        radius_ratio=0.008
    )
    enhanced_importance = enhanced_importance * np.clip(content_guard, 0.0, 1.0)
    enhanced_importance = np.clip(enhanced_importance, 0.0, 1.0)

    return enhanced_importance


def find_light_text_on_color(image, color_importance_map):
    """
    Find light/white text sitting on coloured backgrounds so the cleaner can
    whiten (not darken) it. Returns a float mask (1 = light-on-colour text).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, s, v = cv2.split(hsv)

    colored_background = _soft_ramp(0.45, 0.7, color_importance_map)

    is_light = _soft_ramp(0.65, 0.8, v.astype(np.float32) / 255.0)
    is_low_sat = 1.0 - _soft_ramp(0.08, 0.25, s.astype(np.float32) / 255.0)
    is_white_like = is_light * is_low_sat

    edges = cv2.Canny(gray, 30, 120)
    has_edges = cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (3, 3), 0)

    kernel_size = _scale_blur_window(image.shape, base_ratio=0.01, min_size=5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    colored_background_dilated = cv2.dilate(colored_background, kernel, iterations=1)

    light_text_mask = is_white_like * has_edges * colored_background_dilated
    gray_for_guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    try:
        light_text_mask = _boundary_preserving_blur(light_text_mask, gray_for_guide, radius_ratio=0.02, eps=1e-3)
    except Exception:
        kernel_size = _scale_blur_window(image.shape, base_ratio=0.02, min_size=5)
        light_text_mask = cv2.GaussianBlur(light_text_mask, (kernel_size, kernel_size), 0)
    light_text_mask = scrub_invalid_values(light_text_mask, fill_value=0.0)
    light_text_mask = np.clip(light_text_mask, 0.0, 1.0)

    return light_text_mask


def blend_by_priority(original_image, cleaned_image, gentle_cleaned_image, importance_map):
    """
    Blend three streams by the colour-priority map with soft transitions:

    - priority 0.8-1.0: original (passthrough)
    - priority 0.3-0.8: gentle clean
    - priority 0.0-0.3: full clean
    """
    importance_map = scrub_invalid_values(importance_map, fill_value=0.0)
    importance_map = np.clip(importance_map, 0.0, 1.0)

    gray_for_guide = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    try:
        importance_map_smooth = _boundary_preserving_blur(importance_map, gray_for_guide, radius_ratio=0.03, eps=1e-3)
    except Exception:
        kernel_size = max(7, min(original_image.shape[0], original_image.shape[1]) // 60)
        if kernel_size % 2 == 0:
            kernel_size += 1
        importance_map_smooth = cv2.GaussianBlur(importance_map, (kernel_size, kernel_size), 0)
    importance_map_smooth = scrub_invalid_values(importance_map_smooth, fill_value=0.0)

    # Passthrough weight (0.7 -> 1.0).
    passthrough_weight = guarded_ratio(importance_map_smooth - 0.7, 0.3, epsilon=1e-8, default=0.0)
    passthrough_weight = np.clip(passthrough_weight, 0.0, 1.0)
    passthrough_weight = np.power(passthrough_weight, 0.8)
    passthrough_weight = scrub_invalid_values(passthrough_weight, fill_value=0.0)

    # Gentle weight (peak around mid priority).
    gentle_low = guarded_ratio(importance_map_smooth - 0.2, 0.3, epsilon=1e-8, default=0.0)
    gentle_low = np.clip(gentle_low, 0.0, 1.0)
    gentle_high_val = guarded_ratio(importance_map_smooth - 0.7, 0.3, epsilon=1e-8, default=0.0)
    gentle_high = 1.0 - np.clip(gentle_high_val, 0.0, 1.0)
    gentle_weight = np.minimum(gentle_low, gentle_high)
    gentle_weight = scrub_invalid_values(gentle_weight, fill_value=0.0)

    # Full-clean weight (1.0 -> 0.0).
    full_clean_val = guarded_ratio(importance_map_smooth - 0.2, 0.3, epsilon=1e-8, default=0.0)
    full_clean_weight = 1.0 - np.clip(full_clean_val, 0.0, 1.0)
    full_clean_weight = np.power(full_clean_weight, 0.8)
    full_clean_weight = scrub_invalid_values(full_clean_weight, fill_value=0.0)

    passthrough_weight = cv2.GaussianBlur(passthrough_weight, (11, 11), 0)
    gentle_weight = cv2.GaussianBlur(gentle_weight, (11, 11), 0)
    full_clean_weight = cv2.GaussianBlur(full_clean_weight, (11, 11), 0)

    passthrough_weight = scrub_invalid_values(passthrough_weight, fill_value=0.0)
    gentle_weight = scrub_invalid_values(gentle_weight, fill_value=0.0)
    full_clean_weight = scrub_invalid_values(full_clean_weight, fill_value=0.0)

    # Normalise so the three weights sum to 1.
    total = passthrough_weight + gentle_weight + full_clean_weight
    total = scrub_invalid_values(total, fill_value=1.0)
    total = np.maximum(total, 1e-6)
    passthrough_weight = guarded_ratio(passthrough_weight, total, epsilon=1e-8, default=0.33)
    gentle_weight = guarded_ratio(gentle_weight, total, epsilon=1e-8, default=0.33)
    full_clean_weight = guarded_ratio(full_clean_weight, total, epsilon=1e-8, default=0.34)

    passthrough_3ch = cv2.merge([passthrough_weight, passthrough_weight, passthrough_weight])
    gentle_3ch = cv2.merge([gentle_weight, gentle_weight, gentle_weight])
    full_clean_3ch = cv2.merge([full_clean_weight, full_clean_weight, full_clean_weight])

    orig_float = original_image.astype(np.float32)
    gentle_float = gentle_cleaned_image.astype(np.float32)
    cleaned_float = cleaned_image.astype(np.float32)

    result = (
        orig_float * passthrough_3ch +
        gentle_float * gentle_3ch +
        cleaned_float * full_clean_3ch
    )

    result = scrub_invalid_values(result, fill_value=128.0)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def deepen_ink_strokes(image, max_darkening=85, saturation_threshold=40, color_importance_map=None, dark_mask=None):
    """
    Deepen dark ink/text strokes toward black while protecting coloured
    elements (via the priority map) and excluding large dark regions
    (logos/photos) supplied in `dark_mask`.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    if color_importance_map is not None:
        importance_reduction = np.clip(color_importance_map * 1.25, 0.0, 1.0)

        # Never protect bright paper background even if the map says so.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        brightness_norm = v.astype(np.float32) / 255.0
        sat_norm = s.astype(np.float32) / 255.0

        is_bright_background = (brightness_norm > 0.75).astype(np.float32) * (sat_norm < 0.15).astype(np.float32)
        background_exclusion = 1.0 - is_bright_background * 0.95
        importance_reduction = importance_reduction * background_exclusion

        enhancement_mask = 1.0 - importance_reduction
        very_dark_reduction = np.clip(color_importance_map * 0.9, 0.0, 1.0)
        very_dark_reduction = very_dark_reduction * background_exclusion
    else:
        enhancement_mask = np.ones(image.shape[:2], dtype=np.float32)
        very_dark_reduction = np.ones(image.shape[:2], dtype=np.float32)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    norm_gray = gray / 255.0

    # Only deepen low-saturation ink, never coloured buttons.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    norm_s = s.astype(np.float32) / 255.0
    is_low_saturation_text = (norm_s < 0.25).astype(np.float32)
    is_low_saturation_text = cv2.GaussianBlur(is_low_saturation_text, (5, 5), 0)

    # Exclude big dark regions from ink deepening.
    dark_region_exclusion = np.ones_like(norm_gray, dtype=np.float32)
    if dark_mask is not None:
        dark_mask_norm = (dark_mask.astype(np.float32) / 255.0) if dark_mask.max() > 1.0 else dark_mask.astype(np.float32)
        dark_mask_norm = cv2.GaussianBlur(dark_mask_norm, (9, 9), 0)
        dark_region_exclusion = 1.0 - np.clip(dark_mask_norm * 0.95, 0.0, 1.0)

    # Strokes = dense edges over dark pixels.
    edges_text = cv2.Canny(gray.astype(np.uint8), 50, 170)
    edge_density = cv2.GaussianBlur(edges_text.astype(np.float32) / 255.0, (5, 5), 0)
    stroke_from_edges = _soft_ramp(0.06, 0.18, edge_density)
    darkness_preference = _soft_ramp(0.25, 0.7, 1.0 - norm_gray)
    text_stroke_mask = np.clip(stroke_from_edges * darkness_preference, 0.0, 1.0)
    text_stroke_mask = text_stroke_mask * is_low_saturation_text
    text_stroke_mask = text_stroke_mask * dark_region_exclusion

    dark_mask_curve = np.power(1.0 - norm_gray, 1.5)

    kernel = max(3, (min(image.shape[0], image.shape[1]) // 200) * 2 + 1)
    if kernel % 2 == 0:
        kernel += 1
    kernel = max(3, kernel)
    dark_mask_curve = cv2.GaussianBlur(dark_mask_curve, (kernel, kernel), 0)

    dark_mask_curve = dark_mask_curve * enhancement_mask
    dark_mask_curve = dark_mask_curve * dark_region_exclusion

    adaptive_darkening = dark_mask_curve * max_darkening

    # Graduated targets so already-dark ink reaches near-black.
    target_norm = np.where(
        norm_gray < 0.35,
        0.05,
        np.where(
            norm_gray < 0.65,
            norm_gray * 0.55,
            norm_gray * 0.8
        )
    )
    target_norm = np.clip(target_norm, 0.0, 1.0)
    reduction_needed = np.clip(norm_gray - target_norm, 0.0, 1.0)

    if color_importance_map is not None:
        text_protection = (1.0 - np.clip(color_importance_map * 0.6, 0.0, 0.8)) * background_exclusion
    else:
        text_protection = np.ones_like(norm_gray, dtype=np.float32)

    text_darkening = reduction_needed * 255.0 * text_stroke_mask * text_protection
    text_darkening = scrub_invalid_values(text_darkening, fill_value=0.0)
    adaptive_darkening += text_darkening

    very_dark_mask = (norm_gray < 0.4).astype(np.float32)
    very_dark_mask = cv2.GaussianBlur(very_dark_mask, (5, 5), 0)
    very_dark_mask = very_dark_mask * very_dark_reduction
    very_dark_mask = very_dark_mask * dark_region_exclusion
    very_dark_mask = scrub_invalid_values(very_dark_mask, fill_value=0.0)
    adaptive_darkening += very_dark_mask * 15
    adaptive_darkening = scrub_invalid_values(adaptive_darkening, fill_value=0.0)

    l = scrub_invalid_values(l.astype(np.float32) - adaptive_darkening, fill_value=l.astype(np.float32))
    l = np.clip(l, 0, 255).astype(np.uint8)

    lab_darkened = cv2.merge([l, a, b])
    return cv2.cvtColor(lab_darkened, cv2.COLOR_LAB2BGR)


def sharpen_written_content(image, saturation_threshold=40, color_importance_map=None, white_on_color_mask=None, dark_mask=None):
    """
    Boost and sharpen written content (dark low-saturation text), protecting
    coloured elements and big dark regions, and whitening light text that sits
    on coloured backgrounds instead of darkening it.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    if color_importance_map is not None:
        importance_reduction = np.clip(color_importance_map * 1.25, 0.0, 1.0)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        brightness_norm = v.astype(np.float32) / 255.0
        sat_norm = s.astype(np.float32) / 255.0

        is_bright_background = (brightness_norm > 0.75).astype(np.float32) * (sat_norm < 0.15).astype(np.float32)
        background_exclusion = 1.0 - is_bright_background * 0.95
        importance_reduction = importance_reduction * background_exclusion

        enhancement_mask = 1.0 - importance_reduction
        very_dark_reduction = np.clip(color_importance_map * 0.9, 0.0, 1.0)
        very_dark_reduction = very_dark_reduction * background_exclusion
    else:
        enhancement_mask = np.ones(image.shape[:2], dtype=np.float32)
        very_dark_reduction = np.ones(image.shape[:2], dtype=np.float32)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    norm_s = s.astype(np.float32) / 255.0
    is_grayscale_text = (norm_s < 0.2).astype(np.float32)
    is_grayscale_text = cv2.GaussianBlur(is_grayscale_text, (5, 5), 0)

    dark_region_exclusion = np.ones_like(gray, dtype=np.float32)
    if dark_mask is not None:
        dark_mask_norm = (dark_mask.astype(np.float32) / 255.0) if dark_mask.max() > 1.0 else dark_mask.astype(np.float32)
        dark_mask_norm = cv2.GaussianBlur(dark_mask_norm, (9, 9), 0)
        dark_region_exclusion = 1.0 - np.clip(dark_mask_norm * 0.95, 0.0, 1.0)

    adaptive_thresh = cv2.adaptiveThreshold(
        gray.astype(np.uint8),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)

    text_mask = ((gray < 180) & (edges > 0)).astype(np.float32)
    text_mask = text_mask * is_grayscale_text
    text_mask = cv2.dilate(text_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    text_mask = cv2.GaussianBlur(text_mask, (5, 5), 0)

    text_mask = text_mask * enhancement_mask
    text_mask = text_mask * dark_region_exclusion

    norm_gray = gray / 255.0
    darkness_factor = np.power(1.0 - norm_gray, 2.0)

    combined_mask = text_mask * darkness_factor

    max_boost = 100
    darkening = combined_mask * max_boost

    very_dark = (norm_gray < 0.35).astype(np.float32)
    very_dark = cv2.GaussianBlur(very_dark, (3, 3), 0)
    very_dark = very_dark * very_dark_reduction
    very_dark = very_dark * dark_region_exclusion
    darkening += very_dark * 25

    darkening = scrub_invalid_values(darkening, fill_value=0.0)
    l_enhanced = scrub_invalid_values(l.astype(np.float32) - darkening, fill_value=l.astype(np.float32))
    l_enhanced = np.clip(l_enhanced, 0, 255).astype(np.uint8)

    # Whiten light text on colour instead of darkening it.
    if white_on_color_mask is not None:
        white_text_boost = white_on_color_mask * 30
        white_text_boost = scrub_invalid_values(white_text_boost, fill_value=0.0)
        l_enhanced = scrub_invalid_values(l_enhanced.astype(np.float32) + white_text_boost, fill_value=l_enhanced.astype(np.float32))
        l_enhanced = np.clip(l_enhanced, 0, 255).astype(np.uint8)

    # Unsharp mask over text.
    l_blurred = cv2.GaussianBlur(l_enhanced, (3, 3), 0)
    l_sharpened = cv2.addWeighted(l_enhanced, 1.5, l_blurred, -0.5, 0)
    l_sharpened = np.clip(l_sharpened, 0, 255).astype(np.uint8)

    text_enhancement_mask = (text_mask > 0.1).astype(np.float32)
    text_enhancement_mask = cv2.GaussianBlur(text_enhancement_mask, (3, 3), 0)
    l_sharpened = l_sharpened.astype(np.float32) * text_enhancement_mask + l.astype(np.float32) * (1.0 - text_enhancement_mask)
    l_sharpened = np.clip(l_sharpened, 0, 255).astype(np.uint8)

    # Black-hat to bite dark text a touch further.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    blackhat = cv2.morphologyEx(l_sharpened, cv2.MORPH_BLACKHAT, kernel)
    blackhat_effect = blackhat.astype(np.float32) * text_enhancement_mask * 0.3
    l_final = np.clip(l_sharpened.astype(np.float32) - blackhat_effect, 0, 255).astype(np.uint8)

    final_text_mask = (text_mask > 0.05).astype(np.float32)
    final_text_mask = cv2.GaussianBlur(final_text_mask, (5, 5), 0)

    if color_importance_map is not None:
        colored_mask_final = (color_importance_map > 0.3).astype(np.float32)
    else:
        colored_mask_final = np.zeros(image.shape[:2], dtype=np.float32)
    colored_mask_final = cv2.GaussianBlur(colored_mask_final, (7, 7), 0)

    blend_mask = final_text_mask * (1.0 - colored_mask_final * 0.8)
    l_final = l_final.astype(np.float32) * blend_mask + l.astype(np.float32) * (1.0 - blend_mask)
    l_final = np.clip(l_final, 0, 255).astype(np.uint8)

    lab_enhanced = cv2.merge([l_final, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def _profile_paper_tone(image):
    """
    Profile the paper: find the brightest paper region, pick a white target,
    classify the paper tone (white/cream/yellowed), and derive adaptive
    whitening parameters.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    paper_mask = (
        (v > 150) &
        (s < 30) &
        (l > 120)
    ).astype(np.float32)

    paper_mask = cv2.GaussianBlur(paper_mask, (21, 21), 0)
    paper_mask = (paper_mask > 0.3).astype(np.float32)

    if np.sum(paper_mask) > 1000:
        paper_l_values = l[paper_mask > 0.5]
        if len(paper_l_values) > 0:
            l_p95 = np.percentile(paper_l_values, 95)
            l_p90 = np.percentile(paper_l_values, 90)
            l_p75 = np.percentile(paper_l_values, 75)
            l_mean = np.mean(paper_l_values)
        else:
            l_p95 = np.percentile(l, 95)
            l_p90 = np.percentile(l, 90)
            l_p75 = np.percentile(l, 75)
            l_mean = np.mean(l)
    else:
        l_p95 = np.percentile(l, 95)
        l_p90 = np.percentile(l, 90)
        l_p75 = np.percentile(l, 75)
        l_mean = np.mean(l)

    # Always aim for pure white.
    target_whiteness = 255.0

    paper_a_values = a[paper_mask > 0.5] if np.sum(paper_mask) > 1000 else a.flatten()
    paper_b_values = b[paper_mask > 0.5] if np.sum(paper_mask) > 1000 else b.flatten()

    avg_a = np.mean(paper_a_values)
    avg_b = np.mean(paper_b_values)

    if avg_b > 135:
        if l_mean < 160:
            paper_type = 'yellowed'
        else:
            paper_type = 'cream'
    elif avg_b < 120:
        paper_type = 'white'
    else:
        paper_type = 'white'

    if l_mean < 140:
        background_threshold = 0.50
    elif l_mean < 170:
        background_threshold = 0.55
    elif l_mean < 200:
        background_threshold = 0.60
    else:
        background_threshold = 0.65

    b_deviation = abs(avg_b - 128)
    color_cast_strength = np.clip(b_deviation / 50.0, 0.0, 1.0)

    l_float = l.astype(np.float32)
    distance_to_target = target_whiteness - l_float
    distance_to_target = np.clip(distance_to_target, 0, 255)

    max_distance = target_whiteness - 100
    max_distance = max(max_distance, 50)
    whitening_strength_map = guarded_ratio(distance_to_target, max_distance, epsilon=1.0, default=0.0)
    whitening_strength_map = np.clip(whitening_strength_map, 0.0, 1.0)

    whitening_strength_map = np.power(whitening_strength_map, 0.85)

    return {
        'target_whiteness': target_whiteness,
        'paper_type': paper_type,
        'background_threshold': background_threshold,
        'whitening_strength_map': whitening_strength_map,
        'color_cast_strength': color_cast_strength,
        'avg_b': avg_b,
        'avg_a': avg_a,
        'l_p95': l_p95,
        'l_mean': l_mean
    }


def _map_uneven_lighting(image, window_size=128):
    """
    Map how far each local window's background sits below pure white, so the
    bleach pass can push harder in dark corners / textured paper.
    Returns a 0-1 deviation map (1 = maximum deviation).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height, width = gray.shape

    brightness_map = np.zeros_like(gray)
    weight_map = np.zeros_like(gray)

    stride = window_size // 2
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            y_end = min(y + window_size, height)
            x_end = min(x + window_size, width)
            patch = gray[y:y_end, x:x_end]

            if patch.size == 0:
                continue

            local_bg = np.percentile(patch, 90)  # background, ignoring dark ink
            deviation = 255.0 - local_bg

            brightness_map[y:y_end, x:x_end] += deviation
            weight_map[y:y_end, x:x_end] += 1.0

    weight_map = np.maximum(weight_map, 1e-3)
    brightness_map = guarded_ratio(brightness_map, weight_map, epsilon=1.0, default=0.0)
    brightness_map = scrub_invalid_values(brightness_map, fill_value=0.0)

    kernel_size = window_size // 2
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(15, min(kernel_size, 101))
    brightness_map = cv2.GaussianBlur(brightness_map, (kernel_size, kernel_size), 0)

    max_deviation = np.max(brightness_map)
    if max_deviation > 10:
        deviation_score = brightness_map / max_deviation
    else:
        deviation_score = np.zeros_like(brightness_map)

    return np.clip(deviation_score, 0.0, 1.0)


def _locate_clean_paper(image, gray, norm_gray, color_importance_map=None):
    """
    Locate ONLY pristine background (bright, edge-free, uniform) that is 100%
    safe to force to pure white. Returns a 0-1 mask (1 = clean paper).
    """
    is_bright = (norm_gray > 0.65).astype(np.float32)

    edges = cv2.Canny(gray.astype(np.uint8), 30, 100)
    edges_dilated = cv2.dilate(
        edges,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1
    )
    no_edges = (1.0 - edges_dilated.astype(np.float32) / 255.0)

    kernel = np.ones((15, 15), np.float32) / 225
    mean_local = cv2.filter2D(norm_gray, -1, kernel)
    variance_local = cv2.filter2D((norm_gray - mean_local) ** 2, -1, kernel)
    is_uniform = (variance_local < 0.02).astype(np.float32)

    clean_paper = is_bright * no_edges * is_uniform

    try:
        clean_paper = _boundary_preserving_blur(clean_paper, norm_gray, radius_ratio=0.03, eps=1e-3)
    except Exception:
        blur_size = max(5, min(gray.shape) // 80)
        if blur_size % 2 == 0:
            blur_size += 1
        clean_paper = cv2.GaussianBlur(clean_paper, (blur_size, blur_size), 0)

    clean_paper = _tighten_region_mask(
        clean_paper,
        norm_gray,
        erosion_radius=4,
        edge_stop=0.85,
        radius_ratio=0.008
    )

    try:
        clean_paper = _boundary_preserving_blur(clean_paper, norm_gray, radius_ratio=0.02, eps=1e-3)
    except Exception:
        secondary_blur = max(3, min(gray.shape) // 100)
        if secondary_blur % 2 == 0:
            secondary_blur += 1
        clean_paper = cv2.GaussianBlur(clean_paper, (secondary_blur, secondary_blur), 0)

    clean_paper = _tighten_region_mask(
        clean_paper,
        norm_gray,
        erosion_radius=3,
        edge_stop=0.85,
        radius_ratio=0.008
    )
    clean_paper = np.clip(clean_paper, 0.0, 1.0)

    if color_importance_map is None:
        color_importance_map = build_color_priority_map(image, min_saturation_threshold=30)
    content_guard = _locate_all_content(image, gray, norm_gray, color_importance_map)
    content_guard = _tighten_region_mask(
        content_guard,
        norm_gray,
        erosion_radius=3,
        edge_stop=0.85,
        radius_ratio=0.008
    )
    safety_margin = np.clip(1.0 - content_guard * 0.99, 0.0, 1.0)
    clean_paper = np.clip(clean_paper * safety_margin, 0.0, 1.0)
    clean_paper = _tighten_region_mask(
        clean_paper,
        norm_gray,
        erosion_radius=3,
        edge_stop=0.85,
        radius_ratio=0.008
    )

    return np.clip(clean_paper, 0.0, 1.0)


# Single-slot memo for the shadow mask. Within one page, _locate_all_content
# runs more than once (colour-priority map, graphic-element flagging, ...) with
# the SAME image/gray, and each call recomputes an identical shadow mask via
# heavy Canny/filter2D passes. Caching the last (gray, norm_gray) result skips
# that repeat while returning byte-identical output; different inputs recompute.
_SHADOW_MASK_CACHE = {'key': None, 'mask': None}


def _locate_shadow_patches(image, gray, norm_gray):
    """
    Adaptively find dark background regions (shadows, lighting variation) that
    are NOT content and should be whitened. Thresholds adapt to the image's own
    brightness/saturation/texture distributions. Returns a 0-1 mask.

    Thin memoizing wrapper over `_locate_shadow_patches_impl`: repeat calls with
    the same gray/norm_gray reuse the previous result unchanged.
    """
    gray_c = np.ascontiguousarray(gray)
    norm_c = np.ascontiguousarray(norm_gray)
    key = (
        gray_c.shape,
        gray_c.dtype.str,
        hash(gray_c.tobytes()),
        hash(norm_c.tobytes()),
    )
    if _SHADOW_MASK_CACHE['key'] == key:
        return _SHADOW_MASK_CACHE['mask'].copy()

    mask = _locate_shadow_patches_impl(image, gray, norm_gray)
    _SHADOW_MASK_CACHE['key'] = key
    _SHADOW_MASK_CACHE['mask'] = mask.copy()
    return mask


def _locate_shadow_patches_impl(image, gray, norm_gray):
    """Uncached shadow-mask computation (see `_locate_shadow_patches`)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    norm_s = s.astype(np.float32) / 255.0

    # np.percentile/mean/std flatten internally, so operate on norm_gray directly.
    brightness_p10 = np.percentile(norm_gray, 10)
    brightness_p75 = np.percentile(norm_gray, 75)
    brightness_mean = np.mean(norm_gray)
    brightness_std = np.std(norm_gray)

    saturation_p75 = np.percentile(norm_s, 75)

    kernel = np.ones((21, 21), np.float32) / 441
    mean_local = cv2.filter2D(norm_gray, -1, kernel)
    variance_local = cv2.filter2D((norm_gray - mean_local) ** 2, -1, kernel)
    variance_p75 = np.percentile(variance_local, 75)

    # Adaptive shadow band.
    shadow_lower = np.clip(brightness_p10 * 0.6, 0.08, 0.30)
    shadow_upper = np.clip(brightness_p75 * 1.1, 0.6, 0.90)
    is_shadow_brightness = ((norm_gray > shadow_lower) & (norm_gray < shadow_upper)).astype(np.float32)

    adaptive_sat_threshold = np.clip(saturation_p75 * 1.5, 0.15, 0.35)
    is_low_saturation = (norm_s < adaptive_sat_threshold).astype(np.float32)

    edge_low = max(20, int(brightness_std * 100))
    edge_high = max(50, int(brightness_std * 200))
    edge_low = min(edge_low, 50)
    edge_high = min(edge_high, 150)
    edges = cv2.Canny(gray.astype(np.uint8), edge_low, edge_high)

    dilation_size = max(5, min(gray.shape) // 60)
    if dilation_size % 2 == 0:
        dilation_size += 1
    edges_dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size)), iterations=1)
    has_low_edges = (1.0 - edges_dilated.astype(np.float32) / 255.0)

    adaptive_variance_threshold = np.clip(variance_p75 * 1.2, 0.015, 0.04)
    is_smooth = (variance_local < adaptive_variance_threshold).astype(np.float32)

    image_area = image.shape[0] * image.shape[1]
    area_factor = np.clip(1.0 - (brightness_mean - 0.5) * 0.5, 0.5, 1.5)
    min_area_ratio = 0.003 * area_factor
    min_area = int(min_area_ratio * image_area)
    min_area = max(min_area, 100)

    shadow_candidate = (is_shadow_brightness * is_low_saturation * has_low_edges * is_smooth).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(shadow_candidate, connectivity=8)
    large_shadow_mask = np.zeros_like(shadow_candidate, dtype=np.float32)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            large_shadow_mask[labels == label] = 1.0

    bright_threshold = np.clip(brightness_p75 * 0.95, 0.55, 0.75)
    bright_regions = (norm_gray > bright_threshold).astype(np.uint8) * 255

    bright_dilation_size = max(15, min(gray.shape) // 40)
    if bright_dilation_size % 2 == 0:
        bright_dilation_size += 1
    bright_dilated = cv2.dilate(bright_regions, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bright_dilation_size, bright_dilation_size)), iterations=1)
    near_bright = (bright_dilated > 0).astype(np.float32)

    height, width = gray.shape
    border_thickness = max(min(height, width) // 10, min(height, width) // 15)
    border_mask = np.zeros_like(norm_gray, dtype=np.float32)
    border_mask[0:border_thickness, :] = 1.0
    border_mask[height-border_thickness:height, :] = 1.0
    border_mask[:, 0:border_thickness] = 1.0
    border_mask[:, width-border_thickness:width] = 1.0
    border_blur_size = border_thickness * 2 + 1
    if border_blur_size % 2 == 0:
        border_blur_size += 1
    try:
        border_mask = _boundary_preserving_blur(border_mask, norm_gray, radius_ratio=0.05, eps=1e-3)
    except Exception:
        border_mask = cv2.GaussianBlur(border_mask, (min(15, border_blur_size), min(15, border_blur_size)), 0)
    is_near_border = (border_mask > 0.3).astype(np.float32)

    shadow_mask = (
        is_shadow_brightness *
        is_low_saturation *
        has_low_edges *
        is_smooth *
        large_shadow_mask *
        (near_bright + is_near_border * 0.7)
    )
    shadow_mask = np.clip(shadow_mask, 0.0, 1.0)

    # Extra-aggressive catch for very dark smooth border shadows.
    is_very_dark_shadow = ((norm_gray < 0.35) & (norm_gray > shadow_lower)).astype(np.float32)
    very_dark_smooth = is_very_dark_shadow * is_smooth * is_low_saturation * has_low_edges
    very_dark_near_border = very_dark_smooth * is_near_border * 1.0
    very_dark_near_bright = very_dark_smooth * near_bright * 0.8
    very_dark_shadow_mask = np.maximum(very_dark_near_border, very_dark_near_bright)

    shadow_mask = np.maximum(shadow_mask, very_dark_shadow_mask * 0.9)
    shadow_mask = np.clip(shadow_mask, 0.0, 1.0)

    # Gradient-transition shadows (finger shadow fall-off).
    gradient_kernel = np.array([[-1, 0, 1]], dtype=np.float32)
    grad_x = cv2.filter2D(norm_gray, -1, gradient_kernel)
    grad_y = cv2.filter2D(norm_gray, -1, gradient_kernel.T)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    grad_p25 = np.percentile(gradient_magnitude, 25)
    grad_p75 = np.percentile(gradient_magnitude, 75)
    grad_low = np.clip(grad_p25 * 0.8, 0.03, 0.08)
    grad_high = np.clip(grad_p75 * 1.2, 0.2, 0.4)

    is_smooth_transition = ((gradient_magnitude > grad_low) & (gradient_magnitude < grad_high)).astype(np.float32)
    transition_blur_size = max(15, min(gray.shape) // 40)
    if transition_blur_size % 2 == 0:
        transition_blur_size += 1
    is_smooth_transition = cv2.GaussianBlur(is_smooth_transition, (transition_blur_size, transition_blur_size), 0)

    transition_shadow = (
        is_shadow_brightness *
        is_low_saturation *
        is_smooth_transition *
        (near_bright + is_near_border * 0.7)
    )
    shadow_mask = np.maximum(shadow_mask, transition_shadow * 0.8)

    try:
        shadow_mask = _boundary_preserving_blur(shadow_mask, norm_gray, radius_ratio=0.03, eps=1e-3)
    except Exception:
        kernel_size = max(15, min(gray.shape) // 30)
        if kernel_size % 2 == 0:
            kernel_size += 1
        shadow_mask = cv2.GaussianBlur(shadow_mask, (kernel_size, kernel_size), 0)
    shadow_mask = _tighten_region_mask(shadow_mask, norm_gray, erosion_radius=4, edge_stop=0.85, radius_ratio=0.008)

    return np.clip(shadow_mask, 0.0, 1.0)


def _locate_all_content(image, gray, norm_gray, color_importance_map):
    """
    Detect ALL content that must be preserved (text incl. grey text, images,
    graphics, colour) using OR logic, while explicitly EXCLUDING dark background
    shadows. Returns a 0-1 mask (1 = content to keep).
    """
    shadow_mask = _locate_shadow_patches(image, gray, norm_gray)

    # Strong edges.
    edges = cv2.Canny(gray.astype(np.uint8), 40, 120)
    edges_dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    has_strong_edges = edges_dilated.astype(np.float32) / 255.0

    # Very dark + textured = content (smooth dark = shadow).
    is_very_dark_raw = (norm_gray < 0.4).astype(np.float32)
    kernel_large = np.ones((15, 15), np.float32) / 225
    mean_local_large = cv2.filter2D(norm_gray, -1, kernel_large)
    variance_local_large = cv2.filter2D((norm_gray - mean_local_large) ** 2, -1, kernel_large)
    has_texture = (variance_local_large > 0.015).astype(np.float32)
    is_very_dark = is_very_dark_raw * has_texture

    # Medium-dark + high texture = text.
    kernel = np.ones((9, 9), np.float32) / 81
    mean_local = cv2.filter2D(norm_gray, -1, kernel)
    variance_local = cv2.filter2D((norm_gray - mean_local) ** 2, -1, kernel)
    has_high_texture = (variance_local > 0.02).astype(np.float32)
    is_medium_dark = ((norm_gray > 0.3) & (norm_gray < 0.7)).astype(np.float32)
    textured_content = is_medium_dark * has_high_texture

    # Colour.
    has_color = (color_importance_map > 0.2).astype(np.float32)

    # Bright + textured = graphics on white.
    is_bright = (norm_gray > 0.6).astype(np.float32)
    bright_textured = is_bright * has_high_texture

    content_mask = np.maximum(
        has_strong_edges,
        np.maximum(
            is_very_dark,
            np.maximum(
                textured_content,
                np.maximum(
                    bright_textured,
                    has_color
                )
            )
        )
    )

    # Strip shadows from the protected set.
    content_mask = content_mask * (1.0 - shadow_mask * 0.98)

    # Also strip smooth dark low-saturation regions (shadows).
    is_smooth_dark = ((norm_gray < 0.5) & (variance_local_large < 0.01)).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    norm_s = s.astype(np.float32) / 255.0
    is_low_sat = (norm_s < 0.2).astype(np.float32)

    smooth_dark_shadow = is_smooth_dark * is_low_sat
    content_mask = content_mask * (1.0 - smooth_dark_shadow * 0.95)

    try:
        content_mask = _boundary_preserving_blur(content_mask, norm_gray, radius_ratio=0.03, eps=1e-3)
    except Exception:
        kernel_size = max(9, min(gray.shape) // 50)
        if kernel_size % 2 == 0:
            kernel_size += 1
        content_mask = cv2.GaussianBlur(content_mask, (kernel_size, kernel_size), 0)

    # Erode to shrink the protected zone back onto real content.
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    content_mask = cv2.erode(content_mask, erosion_kernel, iterations=2)

    return np.clip(content_mask, 0.0, 1.0)


def bleach_paper_background(image, whitening_strength=0.25, color_importance_map=None, dark_mask=None, content_protection_mask=None, collect_debug=False):
    """
    Force the paper background to pure white (255) while preserving content.

    Uses the precise masks (colour-priority, dark-region, content-protection)
    to decide what to leave alone, then applies an initial pass, several
    iterative fine-tune passes, a final push, a grey-patch texture cleanup, and
    a colour-cast neutralisation - all restricted to the safe background.

    When `collect_debug` is True, returns (result, debug_snapshots).
    """
    debug_steps = {} if collect_debug else None

    if color_importance_map is None:
        color_importance_map = build_color_priority_map(image, min_saturation_threshold=35)

    deviation_map = _map_uneven_lighting(image, window_size=128)

    # Prefer per-tile targets; fall back to a global profile if they look seamy.
    try:
        local_maps = _derive_local_paper_targets(image, color_importance_map)
        target_map = local_maps['target_map']
        local_strength_map = local_maps['strength_map']
        background_thresh_map = local_maps['background_thresh_map']

        target_diff = np.abs(np.diff(target_map, axis=1))
        if np.max(target_diff) > 50:
            raise ValueError("Local maps have artifacts, using global")
    except (ValueError, Exception):
        analysis_global = _profile_paper_tone(image)
        target_map = np.full((image.shape[0], image.shape[1]), 255.0, dtype=np.float32)
        local_strength_map = analysis_global['whitening_strength_map'] * 3.5
        background_thresh_map = np.full((image.shape[0], image.shape[1]), analysis_global['background_threshold'], dtype=np.float32)

    # Force pure white target everywhere.
    target_map = np.full((image.shape[0], image.shape[1]), 255.0, dtype=np.float32)

    local_strength_map = local_strength_map * (whitening_strength * 4.0)

    # Push harder where the background is uneven/dark.
    adaptive_boost = 1.0 + deviation_map * 2.0
    adaptive_boost = scrub_invalid_values(adaptive_boost, fill_value=1.0)
    local_strength_map = local_strength_map * adaptive_boost

    analysis = _profile_paper_tone(image)
    paper_type = analysis['paper_type']
    color_cast_strength = analysis['color_cast_strength']

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    if collect_debug:
        def _record_mask(name, mask):
            if debug_steps is None:
                return
            mask_vis = np.clip(mask, 0.0, 1.0)
            debug_steps[name] = (mask_vis * 255).astype(np.uint8)

        def _record_l_frame(name, l_channel, a_channel=a, b_channel=b):
            if debug_steps is None:
                return
            lab_frame = cv2.merge([
                np.clip(l_channel, 0, 255).astype(np.uint8),
                np.clip(a_channel, 0, 255).astype(np.uint8),
                np.clip(b_channel, 0, 255).astype(np.uint8)
            ])
            debug_steps[name] = cv2.cvtColor(lab_frame, cv2.COLOR_LAB2BGR)
    else:
        def _record_mask(*args, **kwargs):
            return

        def _record_l_frame(*args, **kwargs):
            return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    norm_gray = gray / 255.0

    # Build the content-protection mask if not provided.
    if content_protection_mask is None:
        high_importance = (color_importance_map > 0.3).astype(np.float32)
        _record_mask('bg_detection_content_protection_(1)_high_importance', high_importance)

        edges = cv2.Canny(gray.astype(np.uint8), 40, 120)
        edges_dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        has_edges = (edges_dilated > 0).astype(np.float32)
        _record_mask('bg_detection_content_protection_(2)_edges_detected', has_edges)

        content_protection_mask = np.maximum(high_importance, has_edges)
        try:
            content_protection_mask = _boundary_preserving_blur(content_protection_mask, norm_gray, radius_ratio=0.02, eps=1e-3)
        except Exception:
            kernel_size = _scale_blur_window(image.shape, base_ratio=0.02, min_size=9)
            content_protection_mask = cv2.GaussianBlur(content_protection_mask, (kernel_size, kernel_size), 0)
        content_protection_mask = np.clip(content_protection_mask, 0.0, 1.0)
        _record_mask('bg_detection_content_protection_(3)_combined', content_protection_mask)

    # Normalise supplied dark mask.
    if dark_mask is not None:
        dark_mask_norm = (dark_mask.astype(np.float32) / 255.0) if dark_mask.max() > 1.0 else dark_mask.astype(np.float32)
    else:
        dark_mask_norm = np.zeros_like(norm_gray, dtype=np.float32)

    # Also detect dark content directly (belt and braces).
    is_very_dark = (norm_gray < 0.35).astype(np.float32)
    _record_mask('bg_detection_dark_protection_(1)_very_dark_pixels', is_very_dark)

    kernel_texture = np.ones((9, 9), np.float32) / 81
    mean_local = cv2.filter2D(norm_gray, -1, kernel_texture)
    variance_local = cv2.filter2D((norm_gray - mean_local) ** 2, -1, kernel_texture)
    has_texture = (variance_local > 0.015).astype(np.float32)

    is_medium_dark = ((norm_gray > 0.25) & (norm_gray < 0.55)).astype(np.float32)
    dark_with_texture = is_medium_dark * has_texture
    _record_mask('bg_detection_dark_protection_(2)_medium_dark_with_texture', dark_with_texture)

    dark_content_detected = np.maximum(is_very_dark, dark_with_texture)
    try:
        dark_content_detected = _boundary_preserving_blur(dark_content_detected, norm_gray, radius_ratio=0.02, eps=1e-3)
    except Exception:
        kernel_size = _scale_blur_window(image.shape, base_ratio=0.02, min_size=9)
        dark_content_detected = cv2.GaussianBlur(dark_content_detected, (kernel_size, kernel_size), 0)
    dark_content_detected = np.clip(dark_content_detected, 0.0, 1.0)
    _record_mask('bg_detection_dark_protection_(3)_dark_content_detected', dark_content_detected)

    dark_protection = np.maximum(dark_mask_norm, dark_content_detected)
    dark_protection = np.clip(dark_protection, 0.0, 1.0)
    _record_mask('bg_detection_dark_protection_(4)_combined', dark_protection)

    if content_protection_mask.max() > 1.0:
        content_protection_mask = content_protection_mask / 255.0
    content_protection_mask = np.clip(content_protection_mask, 0.0, 1.0)

    _record_mask('whiten_content_protection_mask', content_protection_mask)
    _record_mask('whiten_dark_mask', dark_protection)

    # Build the background (whitenable) mask.
    color_protection = np.clip(color_importance_map * 1.2, 0.0, 1.0)
    _record_mask('bg_detection_color_protection', color_protection)

    total_protection = np.maximum(
        content_protection_mask,
        np.maximum(
            color_protection,
            dark_protection
        )
    )
    total_protection = np.clip(total_protection, 0.0, 1.0)
    _record_mask('bg_detection_total_protection', total_protection)

    background_mask = 1.0 - total_protection
    background_mask = background_mask * (1.0 - dark_protection * 0.99)

    try:
        background_mask = _boundary_preserving_blur(background_mask, norm_gray, radius_ratio=0.03, eps=1e-3)
    except Exception:
        kernel_size = _scale_blur_window(image.shape, base_ratio=0.03, min_size=9)
        background_mask = cv2.GaussianBlur(background_mask, (kernel_size, kernel_size), 0)
    background_mask = np.clip(background_mask, 0.0, 1.0)
    _record_mask('bg_detection_background_mask_(1)_after_smooth', background_mask)

    background_mask = _tighten_region_mask(
        background_mask,
        norm_gray,
        erosion_radius=2,
        edge_stop=0.8,
        radius_ratio=0.01
    )
    _record_mask('bg_detection_background_mask_(2)_after_refine', background_mask)

    background_mask = background_mask * (1.0 - total_protection * 0.99)
    background_mask = np.clip(background_mask, 0.0, 1.0)

    # Safe background = background minus dark regions (used in every pass).
    safe_background_mask = background_mask * (1.0 - dark_protection * 0.99)
    safe_background_mask = np.clip(safe_background_mask, 0.0, 1.0)
    _record_mask('bg_detection_background_mask_(3)_final', safe_background_mask)

    _record_mask('whiten_background_mask', 1.0 - background_mask)

    # --- Initial whitening pass ------------------------------------------
    l_float = l.astype(np.float32)
    distance_to_target = np.clip(target_map - l_float, 0, 255)
    max_distance = np.clip(target_map - 50, 100, 255)
    distance_normalized = guarded_ratio(distance_to_target, max_distance, epsilon=1.0, default=0.0)
    distance_normalized = np.clip(distance_normalized, 0.0, 1.0)
    whitening_curve = np.power(distance_normalized, 0.7)
    whitening_curve = scrub_invalid_values(whitening_curve, fill_value=0.0)

    effective_strength = np.clip(local_strength_map * (1.0 + color_cast_strength * 0.4), 0.5, 4.0)
    effective_strength = scrub_invalid_values(effective_strength, fill_value=1.0)

    whitening_amount = distance_to_target * effective_strength * safe_background_mask * (0.9 + whitening_curve * 0.7)
    whitening_amount = scrub_invalid_values(whitening_amount, fill_value=0.0)
    whitening_amount = whitening_amount * (1.0 - dark_protection * 0.99)
    whitening_amount = scrub_invalid_values(whitening_amount, fill_value=0.0)

    l_whitened = scrub_invalid_values(l_float + whitening_amount, fill_value=l_float)
    l_whitened = np.clip(l_whitened, 0, 255).astype(np.uint8)
    _record_l_frame('whiten_initial_pass', l_whitened)

    # --- Iterative fine-tuning -------------------------------------------
    l_whitened_float = l_whitened.astype(np.float32)
    iter_kernel_size = _scale_blur_window(image.shape, base_ratio=0.02, min_size=11)

    # Protect thin text edges more strongly.
    thin_text_edges = cv2.Canny(l_whitened.astype(np.uint8), 30, 100)
    thin_text_mask = cv2.dilate(thin_text_edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)
    thin_text_protection = (thin_text_mask > 0).astype(np.float32)
    try:
        thin_text_protection = _boundary_preserving_blur(thin_text_protection, norm_gray, radius_ratio=0.01, eps=1e-3)
    except Exception:
        thin_text_protection = cv2.GaussianBlur(thin_text_protection, (3, 3), 0)
    thin_text_protection = np.clip(thin_text_protection, 0.0, 1.0)

    enhanced_total_protection = np.maximum(total_protection, thin_text_protection * 0.9)
    enhanced_total_protection = np.clip(enhanced_total_protection, 0.0, 1.0)

    for iteration in range(4):
        remaining_distance = np.clip(target_map - l_whitened_float, 0, 255)

        needs_whitening = (remaining_distance > 2.0).astype(np.float32)
        try:
            needs_whitening = _boundary_preserving_blur(needs_whitening, norm_gray, radius_ratio=0.015, eps=1e-3)
        except Exception:
            needs_whitening = cv2.GaussianBlur(needs_whitening, (min(5, iter_kernel_size), min(5, iter_kernel_size)), 0)

        iteration_strength = 0.8 - (iteration * 0.1)

        content_protection = enhanced_total_protection * 0.99
        needs_whitening = needs_whitening * (1.0 - content_protection)
        needs_whitening = needs_whitening * (1.0 - dark_protection * 0.99)

        fine_tune_amount = remaining_distance * iteration_strength * needs_whitening * safe_background_mask
        fine_tune_amount = scrub_invalid_values(fine_tune_amount, fill_value=0.0)

        fine_tune_amount = fine_tune_amount * (1.0 - dark_protection * 0.99)
        fine_tune_amount = fine_tune_amount * (1.0 - thin_text_protection * 0.95)
        fine_tune_amount = scrub_invalid_values(fine_tune_amount, fill_value=0.0)

        l_whitened_float = scrub_invalid_values(l_whitened_float + fine_tune_amount, fill_value=l_whitened_float)
        l_whitened_float = np.clip(l_whitened_float, 0, 255)

    _record_l_frame('whiten_iterative_refine', l_whitened_float)

    # --- Final push on near-white pixels ---------------------------------
    very_close_mask = (l_whitened_float > 240).astype(np.float32)
    try:
        very_close_mask = _boundary_preserving_blur(very_close_mask, norm_gray, radius_ratio=0.015, eps=1e-3)
    except Exception:
        very_close_mask = cv2.GaussianBlur(very_close_mask, (min(5, iter_kernel_size), min(5, iter_kernel_size)), 0)
    very_close_mask = very_close_mask * (1.0 - enhanced_total_protection * 0.99)
    very_close_mask = very_close_mask * (1.0 - dark_protection * 0.99)
    very_close_mask = very_close_mask * (1.0 - thin_text_protection * 0.95)
    final_push = (255.0 - l_whitened_float) * very_close_mask * safe_background_mask * 0.8
    final_push = scrub_invalid_values(final_push, fill_value=0.0)
    l_whitened_float = scrub_invalid_values(l_whitened_float + final_push, fill_value=l_whitened_float)

    l_whitened = np.clip(l_whitened_float, 0, 255).astype(np.uint8)

    # --- Grey-patch texture cleanup --------------------------------------
    grey_patches = ((l_whitened > 180) & (l_whitened < 252)).astype(np.float32)
    try:
        grey_patches = _boundary_preserving_blur(grey_patches, norm_gray, radius_ratio=0.02, eps=1e-3)
    except Exception:
        grey_patches = cv2.GaussianBlur(grey_patches, (11, 11), 0)

    grey_patches = grey_patches * (1.0 - enhanced_total_protection * 0.99)
    grey_patches = grey_patches * (1.0 - dark_protection * 0.99)
    grey_patches = grey_patches * (1.0 - thin_text_protection * 0.95)
    grey_patches = grey_patches * safe_background_mask

    final_push_texture = (255.0 - l_whitened.astype(np.float32)) * grey_patches * 0.9
    final_push_texture = scrub_invalid_values(final_push_texture, fill_value=0.0)
    l_whitened = np.clip(l_whitened.astype(np.float32) + final_push_texture, 0, 255).astype(np.uint8)
    _record_l_frame('whiten_texture_cleanup', l_whitened)

    # --- Colour-cast neutralisation on background ------------------------
    a_float = a.astype(np.float32)
    b_float = b.astype(np.float32)

    color_neutralize_mask = background_mask * (1.0 - total_protection * 0.95)
    color_neutralize_mask = np.clip(color_neutralize_mask, 0.0, 1.0)

    try:
        color_neutralize_mask = _boundary_preserving_blur(color_neutralize_mask, norm_gray, radius_ratio=0.02, eps=1e-3)
    except Exception:
        kernel_size = _scale_blur_window(image.shape, base_ratio=0.02, min_size=9)
        color_neutralize_mask = cv2.GaussianBlur(color_neutralize_mask, (kernel_size, kernel_size), 0)
    color_neutralize_mask = np.clip(color_neutralize_mask, 0.0, 1.0)

    _record_mask('whiten_color_neutralize_mask', 1.0 - color_neutralize_mask)

    a_float = a.astype(np.float32)
    b_float = b.astype(np.float32)
    if np.any(color_neutralize_mask > 1e-3):
        mask_strength = color_neutralize_mask * 0.85
        a_target = 128.0
        if paper_type == 'yellowed':
            b_target = 125.5
        elif paper_type == 'cream':
            b_target = 127.5
        else:
            b_target = 128.0

        a_float = a_float * (1.0 - mask_strength) + a_target * mask_strength
        b_float = b_float * (1.0 - mask_strength) + b_target * mask_strength

    a = np.clip(a_float, 0, 255).astype(np.uint8)
    b = np.clip(b_float, 0, 255).astype(np.uint8)

    lab_whitened = cv2.merge([l_whitened, a, b])
    result = cv2.cvtColor(lab_whitened, cv2.COLOR_LAB2BGR)

    result = scrub_invalid_values(result.astype(np.float32), fill_value=128.0)
    result = np.clip(result, 0, 255).astype(np.uint8)

    if collect_debug:
        debug_steps['whiten_final'] = result.copy()
        return result, debug_steps

    return result


def boost_content_legibility(image, color_importance_map=None, dark_mask=None, content_protection_mask=None):
    """
    After bleaching, make grayscale text darker/crisper against the white page,
    while leaving coloured content and big dark regions (logos/photos) alone.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    norm_gray = gray / 255.0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    norm_s = s.astype(np.float32) / 255.0
    is_grayscale_content = (norm_s < 0.25).astype(np.float32)
    is_grayscale_content = cv2.GaussianBlur(is_grayscale_content, (5, 5), 0)

    dark_region_exclusion = np.ones_like(norm_gray, dtype=np.float32)
    if dark_mask is not None:
        dark_mask_norm = (dark_mask.astype(np.float32) / 255.0) if dark_mask.max() > 1.0 else dark_mask.astype(np.float32)
        dark_mask_norm = cv2.GaussianBlur(dark_mask_norm, (9, 9), 0)
        dark_region_exclusion = 1.0 - np.clip(dark_mask_norm * 0.95, 0.0, 1.0)

    if content_protection_mask is None:
        edges = cv2.Canny(gray.astype(np.uint8), 40, 120)
        edges_dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        has_edges = (edges_dilated > 0).astype(np.float32)

        if color_importance_map is not None:
            high_importance = (color_importance_map > 0.3).astype(np.float32)
            content_mask = np.maximum(high_importance, has_edges)
        else:
            content_mask = has_edges
    else:
        content_mask = content_protection_mask.copy()
        if content_mask.max() > 1.0:
            content_mask = content_mask / 255.0
        content_mask = np.clip(content_mask, 0.0, 1.0)

    is_very_dark = (norm_gray < 0.4).astype(np.float32)
    is_medium_dark = ((norm_gray > 0.2) & (norm_gray < 0.6)).astype(np.float32)

    all_content = np.maximum(
        content_mask,
        np.maximum(is_very_dark, is_medium_dark * 0.7)
    )
    all_content = all_content * is_grayscale_content
    all_content = np.clip(all_content, 0.0, 1.0)
    all_content = all_content * dark_region_exclusion

    try:
        all_content = _boundary_preserving_blur(all_content, norm_gray, radius_ratio=0.02, eps=1e-3)
    except Exception:
        kernel_size = _scale_blur_window(image.shape, base_ratio=0.02, min_size=9)
        all_content = cv2.GaussianBlur(all_content, (kernel_size, kernel_size), 0)
    all_content = np.clip(all_content, 0.0, 1.0)

    very_dark_mask = (norm_gray < 0.25).astype(np.float32)
    medium_dark_mask = ((norm_gray >= 0.25) & (norm_gray < 0.5)).astype(np.float32)
    light_content_mask = ((norm_gray >= 0.5) & (norm_gray < 0.7)).astype(np.float32)

    very_dark_mask = very_dark_mask * dark_region_exclusion
    medium_dark_mask = medium_dark_mask * dark_region_exclusion
    light_content_mask = light_content_mask * dark_region_exclusion

    very_dark_darkening = very_dark_mask * 35
    medium_dark_darkening = medium_dark_mask * 20
    light_darkening = light_content_mask * 8

    darkening_mask = (very_dark_darkening + medium_dark_darkening + light_darkening) * all_content
    darkening_mask = cv2.GaussianBlur(darkening_mask, (5, 5), 0)

    l_float = l.astype(np.float32)
    l_darkened = l_float - darkening_mask
    l_darkened = np.clip(l_darkened, 0, 255).astype(np.uint8)

    l_contrast = l_darkened.astype(np.float32)

    dark_content_boost = (norm_gray < 0.5).astype(np.float32) * all_content * dark_region_exclusion * 0.3
    light_content_boost = (norm_gray >= 0.5).astype(np.float32) * all_content * dark_region_exclusion * 0.15
    total_boost = dark_content_boost + light_content_boost

    mean_l = np.mean(l_contrast)
    darker_than_mean = (l_contrast < mean_l).astype(np.float32)
    contrast_adjustment = (l_contrast - mean_l) * total_boost
    contrast_adjustment = contrast_adjustment * (1.0 + darker_than_mean * 0.5)

    l_contrast = l_contrast + contrast_adjustment
    l_contrast = np.clip(l_contrast, 0, 255).astype(np.uint8)

    l_blurred = cv2.GaussianBlur(l_contrast, (3, 3), 0)
    unsharp = l_contrast.astype(np.float32) - l_blurred.astype(np.float32)
    sharpening_strength = all_content * dark_region_exclusion * 0.3
    l_sharpened = l_contrast.astype(np.float32) + unsharp * sharpening_strength
    l_sharpened = np.clip(l_sharpened, 0, 255).astype(np.uint8)

    blend_mask = all_content
    l_final = l_sharpened.astype(np.float32) * blend_mask + l.astype(np.float32) * (1.0 - blend_mask)
    l_final = np.clip(l_final, 0, 255).astype(np.uint8)

    lab_enhanced = cv2.merge([l_final, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def revive_original_colors(image, color_importance_map=None, saturation_boost=1.4, vibrancy_boost=1.2):
    """
    Re-energise the page's genuine colours (boost saturation in HSV and vibrancy
    in LAB) only where the priority map says there is real colour, leaving the
    white/neutral background untouched.
    """
    if color_importance_map is None:
        color_importance_map = build_color_priority_map(image, min_saturation_threshold=25)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    color_boost_mask = _soft_ramp(0.15, 0.5, color_importance_map)

    # HSV saturation boost.
    s_float = s.astype(np.float32)
    has_color = (s_float > 20).astype(np.float32)

    saturation_curve = np.power(s_float / 255.0, 0.7)
    saturation_factor = 1.0 + (saturation_boost - 1.0) * color_boost_mask * saturation_curve * has_color

    s_boosted = s_float * saturation_factor
    s_boosted = np.clip(s_boosted, 0, 255).astype(np.uint8)

    # LAB vibrancy boost (push a/b away from neutral).
    a_float = a.astype(np.float32)
    b_float = b.astype(np.float32)

    a_distance = a_float - 128.0
    b_distance = b_float - 128.0

    a_boosted = 128.0 + a_distance * vibrancy_boost * color_boost_mask
    b_boosted = 128.0 + b_distance * vibrancy_boost * color_boost_mask

    a_boosted = np.clip(a_boosted, 0, 255).astype(np.uint8)
    b_boosted = np.clip(b_boosted, 0, 255).astype(np.uint8)

    blend_weight = color_boost_mask

    s_final = s.astype(np.float32) * (1.0 - blend_weight) + s_boosted.astype(np.float32) * blend_weight
    s_final = np.clip(s_final, 0, 255).astype(np.uint8)

    a_final = a.astype(np.float32) * (1.0 - blend_weight) + a_boosted.astype(np.float32) * blend_weight
    b_final = b.astype(np.float32) * (1.0 - blend_weight) + b_boosted.astype(np.float32) * blend_weight
    a_final = np.clip(a_final, 0, 255).astype(np.uint8)
    b_final = np.clip(b_final, 0, 255).astype(np.uint8)

    hsv_boosted = cv2.merge([h, s_final, v])
    result_hsv = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)

    lab_boosted = cv2.merge([l, a_final, b_final])
    result_lab = cv2.cvtColor(lab_boosted, cv2.COLOR_LAB2BGR)

    hsv_weight = 0.6
    lab_weight = 0.4

    result = result_hsv.astype(np.float32) * hsv_weight + result_lab.astype(np.float32) * lab_weight
    result = np.clip(result, 0, 255).astype(np.uint8)

    final_blend = image.astype(np.float32) * (1.0 - color_boost_mask[..., None]) + result.astype(np.float32) * color_boost_mask[..., None]
    result = np.clip(final_blend, 0, 255).astype(np.uint8)

    return result
