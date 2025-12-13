//! Performance metrics for debug overlay
//!
//! Tracks FPS, latencies, and per-arm data for real-time display.

use wasm_bindgen::prelude::*;
use std::cell::RefCell;

/// Performance metrics storage
struct Metrics {
    /// Frame timing
    fps: f32,
    frame_time_ms: f32,
    
    /// Component latencies
    mediapipe_latency_ms: f32,
    onnx_latency_ms: f32,
    physics_time_ms: f32,
    
    /// Per-arm data
    left_depth: f32,
    left_angle: f32,
    left_velocity: f32,
    right_depth: f32,
    right_angle: f32,
    right_velocity: f32,
    
    /// Depth validity
    left_valid: bool,
    right_valid: bool,
}

impl Default for Metrics {
    fn default() -> Self {
        Self {
            fps: 0.0,
            frame_time_ms: 0.0,
            mediapipe_latency_ms: 0.0,
            onnx_latency_ms: 0.0,
            physics_time_ms: 0.0,
            left_depth: 0.0,
            left_angle: 180.0,
            left_velocity: 0.0,
            right_depth: 0.0,
            right_angle: 180.0,
            right_velocity: 0.0,
            left_valid: false,
            right_valid: false,
        }
    }
}

thread_local! {
    static METRICS: RefCell<Metrics> = RefCell::new(Metrics::default());
}

// ============================================================================
// WASM ENTRY POINTS
// ============================================================================

/// Set frame timing (called from JS each frame)
#[wasm_bindgen]
pub fn set_frame_metrics(fps: f32, frame_time_ms: f32) {
    // Sanity check: ignore invalid times (tab backgrounded or first frame)
    if frame_time_ms < 1.0 || frame_time_ms > 200.0 {
        return;
    }
    
    METRICS.with(|m| {
        let mut metrics = m.borrow_mut();
        // Exponential moving average for smooth display
        metrics.fps = metrics.fps * 0.9 + fps * 0.1;
        metrics.frame_time_ms = metrics.frame_time_ms * 0.9 + frame_time_ms * 0.1;
    });
}

/// Set MediaPipe detection latency
#[wasm_bindgen]
pub fn set_mediapipe_latency(ms: f32) {
    METRICS.with(|m| {
        let mut metrics = m.borrow_mut();
        metrics.mediapipe_latency_ms = metrics.mediapipe_latency_ms * 0.9 + ms * 0.1;
    });
}

/// Set ONNX inference latency
#[wasm_bindgen]
pub fn set_onnx_latency(ms: f32) {
    METRICS.with(|m| {
        let mut metrics = m.borrow_mut();
        metrics.onnx_latency_ms = metrics.onnx_latency_ms * 0.9 + ms * 0.1;
    });
}

/// Set physics tick time
#[wasm_bindgen]
pub fn set_physics_time(ms: f32) {
    METRICS.with(|m| {
        let mut metrics = m.borrow_mut();
        metrics.physics_time_ms = metrics.physics_time_ms * 0.9 + ms * 0.1;
    });
}

/// Update arm metrics from bridge data
pub fn update_arm_metrics(
    left_depth: f32, left_angle: f32, left_velocity: f32, left_valid: bool,
    right_depth: f32, right_angle: f32, right_velocity: f32, right_valid: bool,
) {
    METRICS.with(|m| {
        let mut metrics = m.borrow_mut();
        metrics.left_depth = left_depth;
        metrics.left_angle = left_angle;
        metrics.left_velocity = left_velocity;
        metrics.left_valid = left_valid;
        metrics.right_depth = right_depth;
        metrics.right_angle = right_angle;
        metrics.right_velocity = right_velocity;
        metrics.right_valid = right_valid;
    });
}

/// Get formatted overlay text (called from JS to update HTML)
#[wasm_bindgen]
pub fn get_debug_overlay_text() -> String {
    METRICS.with(|m| {
        let metrics = m.borrow();
        format!(
            "FPS: {:.0} | Frame: {:.1}ms\n\
             MediaPipe: {:.0}ms | ONNX: {:.0}ms\n\
             Physics: {:.1}ms\n\
             L: {:.0}%{} {:.0}° v{:.2}\n\
             R: {:.0}%{} {:.0}° v{:.2}",
            metrics.fps, metrics.frame_time_ms,
            metrics.mediapipe_latency_ms, metrics.onnx_latency_ms,
            metrics.physics_time_ms,
            metrics.left_depth,
            if metrics.left_valid { "✓" } else { "✗" },
            metrics.left_angle,
            metrics.left_velocity,
            metrics.right_depth,
            if metrics.right_valid { "✓" } else { "✗" },
            metrics.right_angle,
            metrics.right_velocity,
        )
    })
}
