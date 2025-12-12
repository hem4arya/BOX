//! Skeleton rendering - draws landmarks and bone connections

use super::state::GPU_STATE;
use super::shapes::{Vertex, create_circle_vertices, create_line_vertices};
use crate::bridge;

/// Colors for different landmark types
mod colors {
    pub const RED: [f32; 4] = [1.0, 0.2, 0.2, 1.0];      // Key landmarks
    pub const YELLOW: [f32; 4] = [1.0, 0.9, 0.2, 1.0];   // Wrists (punch detection)
    pub const CYAN: [f32; 4] = [0.2, 0.9, 0.9, 0.7];     // Skeleton lines
    pub const BACKGROUND: wgpu::Color = wgpu::Color {
        r: 0.102, g: 0.102, b: 0.180, a: 1.0  // #1a1a2e
    };
}

/// Convert normalized landmark (0-1) to clip space (-1 to 1), flip Y
fn to_clip_space(x: f32, y: f32) -> (f32, f32) {
    (x * 2.0 - 1.0, -(y * 2.0 - 1.0))
}

/// Build vertex data for all skeleton lines
fn build_skeleton_vertices(landmarks: &[bridge::Landmark; 33]) -> Vec<Vertex> {
    let mut vertices = Vec::new();
    
    for (start_idx, end_idx) in bridge::ARM_SKELETON.iter() {
        let start = landmarks[*start_idx];
        let end = landmarks[*end_idx];
        
        let (x1, y1) = to_clip_space(start.x, start.y);
        let (x2, y2) = to_clip_space(end.x, end.y);
        
        vertices.extend(create_line_vertices(x1, y1, x2, y2, 0.008, colors::CYAN));
    }
    
    vertices
}

/// Build vertex data for key landmark dots
fn build_landmark_vertices(landmarks: &[bridge::Landmark; 33]) -> Vec<Vertex> {
    let mut vertices = Vec::new();
    
    for &idx in bridge::KEY_LANDMARKS.iter() {
        let lm = landmarks[idx];
        let (x, y) = to_clip_space(lm.x, lm.y);
        
        // Wrists are yellow and larger (punch detection points)
        let (color, radius) = if idx == bridge::LEFT_WRIST || idx == bridge::RIGHT_WRIST {
            (colors::YELLOW, 0.025)
        } else {
            (colors::RED, 0.015)
        };
        
        vertices.extend(create_circle_vertices(x, y, radius, color, 16));
    }
    
    vertices
}

/// Render one frame with current landmarks
pub fn render_frame() {
    GPU_STATE.with(|state_cell| {
        let state_ref = state_cell.borrow();
        let state = match state_ref.as_ref() {
            Some(s) => s,
            None => return,
        };

        // Build vertices from current landmarks
        let mut vertices: Vec<Vertex> = Vec::new();
        
        if let Some(landmarks) = bridge::get_all_landmarks() {
            // Lines first (so dots appear on top)
            vertices.extend(build_skeleton_vertices(&landmarks));
            vertices.extend(build_landmark_vertices(&landmarks));
        }

        // Get surface texture
        let output = match state.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = state.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Render Encoder") }
        );

        if !vertices.is_empty() {
            state.queue.write_buffer(
                &state.vertex_buffer,
                0,
                bytemuck::cast_slice(&vertices),
            );
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
