//! Skeleton rendering - draws landmarks, bones, depth bars, and predicted positions

use super::state::GPU_STATE;
use super::shapes::{Vertex, create_circle_vertices, create_line_vertices};
use crate::bridge;

/// Colors for different visualization elements
mod colors {
    /// Raw MediaPipe landmarks (lagged)
    pub const RED: [f32; 4] = [1.0, 0.2, 0.2, 1.0];
    /// Smoothed wrist positions (jitter-free)
    pub const CYAN_BRIGHT: [f32; 4] = [0.3, 1.0, 1.0, 1.0];
    /// Wrists from raw data
    pub const YELLOW: [f32; 4] = [1.0, 0.9, 0.2, 1.0];
    /// Skeleton lines
    pub const CYAN: [f32; 4] = [0.2, 0.9, 0.9, 0.7];
    /// Depth bar background
    pub const GRAY: [f32; 4] = [0.3, 0.3, 0.3, 0.8];
    /// Depth bar fill (valid punch)
    pub const GREEN: [f32; 4] = [0.2, 0.9, 0.3, 0.9];
    /// Depth bar fill (invalid - vertical)
    pub const ORANGE: [f32; 4] = [1.0, 0.6, 0.2, 0.9];
    /// Hand landmark (wrist)
    pub const MAGENTA: [f32; 4] = [1.0, 0.3, 1.0, 1.0];
    /// Hand fingertips
    pub const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
    /// Hand joints
    pub const LIME: [f32; 4] = [0.5, 1.0, 0.3, 1.0];
    /// Hand bones
    pub const PINK: [f32; 4] = [1.0, 0.5, 0.7, 0.9];
    /// Background
    pub const BACKGROUND: wgpu::Color = wgpu::Color {
        r: 0.102, g: 0.102, b: 0.180, a: 1.0
    };
}

fn to_clip_space(x: f32, y: f32) -> (f32, f32) {
    (x * 2.0 - 1.0, -(y * 2.0 - 1.0))
}

// ============================================================================
// HAND SKELETON RENDERING
// ============================================================================

/// Build vertices for a single hand (21 landmarks + bone connections)
fn build_hand_vertices(hand: &bridge::HandData) -> Vec<Vertex> {
    let mut vertices = Vec::new();
    
    if !hand.valid {
        return vertices;
    }
    
    // Draw bone connections first (behind joints)
    for (start_idx, end_idx) in bridge::HAND_SKELETON.iter() {
        let start = &hand.landmarks[*start_idx];
        let end = &hand.landmarks[*end_idx];
        
        // Skip invalid landmarks
        if start.x < 0.001 && start.y < 0.001 {
            continue;
        }
        if end.x < 0.001 && end.y < 0.001 {
            continue;
        }
        
        let (x1, y1) = to_clip_space(start.x, start.y);
        let (x2, y2) = to_clip_space(end.x, end.y);
        vertices.extend(create_line_vertices(x1, y1, x2, y2, 0.004, colors::PINK));
    }
    
    // Draw joint circles (on top of bones)
    for (i, lm) in hand.landmarks.iter().enumerate() {
        if lm.x < 0.001 && lm.y < 0.001 {
            continue;
        }
        
        let (x, y) = to_clip_space(lm.x, lm.y);
        
        // Color and size by joint type
        let (color, radius) = match i {
            0 => (colors::MAGENTA, 0.018),  // Wrist - large magenta
            4 | 8 | 12 | 16 | 20 => (colors::WHITE, 0.012),  // Fingertips - white
            _ => (colors::LIME, 0.008),  // Other joints - small lime
        };
        
        vertices.extend(create_circle_vertices(x, y, radius, color, 12));
    }
    
    vertices
}

// ============================================================================
// POSE SKELETON RENDERING (Legacy - for fallback)
// ============================================================================

fn build_skeleton_vertices(landmarks: &[bridge::Landmark; 33]) -> Vec<Vertex> {
    let mut vertices = Vec::new();
    for (start_idx, end_idx) in bridge::ARM_SKELETON.iter() {
        let start = landmarks[*start_idx];
        let end = landmarks[*end_idx];
        let (x1, y1) = to_clip_space(start.x, start.y);
        let (x2, y2) = to_clip_space(end.x, end.y);
        vertices.extend(create_line_vertices(x1, y1, x2, y2, 0.006, colors::CYAN));
    }
    vertices
}

fn build_raw_landmark_vertices(landmarks: &[bridge::Landmark; 33]) -> Vec<Vertex> {
    let mut vertices = Vec::new();
    for &idx in bridge::KEY_LANDMARKS.iter() {
        let lm = landmarks[idx];
        let (x, y) = to_clip_space(lm.x, lm.y);
        let (color, radius) = if idx == bridge::LEFT_WRIST || idx == bridge::RIGHT_WRIST {
            (colors::YELLOW, 0.015)
        } else {
            (colors::RED, 0.010)
        };
        vertices.extend(create_circle_vertices(x, y, radius, color, 12));
    }
    vertices
}

fn build_smoothed_vertices(smoothed_wrists: [(f32, f32); 2]) -> Vec<Vertex> {
    let mut vertices = Vec::new();
    for (px, py) in smoothed_wrists.iter() {
        let (x, y) = to_clip_space(*px, *py);
        vertices.extend(create_circle_vertices(x, y, 0.028, colors::CYAN_BRIGHT, 16));
    }
    vertices
}

fn create_rect_vertices(x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) -> Vec<Vertex> {
    vec![
        Vertex { position: [x, y], color },
        Vertex { position: [x + w, y], color },
        Vertex { position: [x + w, y + h], color },
        Vertex { position: [x, y], color },
        Vertex { position: [x + w, y + h], color },
        Vertex { position: [x, y + h], color },
    ]
}

// ============================================================================
// MAIN RENDER
// ============================================================================

pub fn render_frame() {
    GPU_STATE.with(|state_cell| {
        let state_ref = state_cell.borrow();
        let state = match state_ref.as_ref() {
            Some(s) => s,
            None => return,
        };

        let mut vertices: Vec<Vertex> = Vec::new();
        
        // Try to render HAND landmarks first (if available)
        if let Some((hands, num_hands)) = bridge::get_hand_data() {
            for i in 0..num_hands {
                vertices.extend(build_hand_vertices(&hands[i]));
            }
        } else {
            // Fallback to POSE landmarks (legacy)
            let (_, _, left_depth, right_depth, _, _) = bridge::get_debug_info();
            let (left_valid, right_valid, _, _) = bridge::get_depth_validity();
            
            if let Some(landmarks) = bridge::get_all_landmarks() {
                vertices.extend(build_skeleton_vertices(&landmarks));
                vertices.extend(build_raw_landmark_vertices(&landmarks));
            }
            
            // Draw EXTRAPOLATED wrists
            let extrap = bridge::get_extrapolated_wrists();
            if extrap.len() == 4 {
                let left = (extrap[0], extrap[1]);
                let right = (extrap[2], extrap[3]);
                vertices.extend(build_smoothed_vertices([left, right]));
            }
        }

        let output = match state.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = state.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Render Encoder") }
        );

        if !vertices.is_empty() {
            state.queue.write_buffer(&state.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Skeleton Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(colors::BACKGROUND),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            if !vertices.is_empty() {
                pass.set_pipeline(&state.render_pipeline);
                pass.set_vertex_buffer(0, state.vertex_buffer.slice(..));
                pass.draw(0..vertices.len() as u32, 0..1);
            }
        }

        state.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    });
}
