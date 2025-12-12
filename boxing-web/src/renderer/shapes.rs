//! Shape primitives - vertices for circles and lines

/// Vertex structure for rendering colored shapes
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 2],
    pub color: [f32; 4],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
        0 => Float32x2,
        1 => Float32x4
    ];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

/// Generate vertices for a filled circle (triangle fan)
pub fn create_circle_vertices(
    cx: f32, 
    cy: f32, 
    radius: f32, 
    color: [f32; 4], 
    segments: u32
) -> Vec<Vertex> {
    let mut vertices = Vec::with_capacity((segments * 3) as usize);
    
    for i in 0..segments {
        let angle1 = (i as f32 / segments as f32) * std::f32::consts::TAU;
        let angle2 = ((i + 1) as f32 / segments as f32) * std::f32::consts::TAU;
        
        vertices.push(Vertex { position: [cx, cy], color });
        vertices.push(Vertex { 
            position: [cx + radius * angle1.cos(), cy + radius * angle1.sin()], 
            color 
        });
        vertices.push(Vertex { 
            position: [cx + radius * angle2.cos(), cy + radius * angle2.sin()], 
            color 
        });
    }
    
    vertices
}

/// Generate vertices for a line segment (rendered as thin quad)
pub fn create_line_vertices(
    x1: f32, y1: f32, 
    x2: f32, y2: f32, 
    width: f32, 
    color: [f32; 4]
) -> Vec<Vertex> {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len = (dx * dx + dy * dy).sqrt();
    
    if len < 0.001 { return vec![]; }
    
    // Perpendicular direction for line thickness
    let px = -dy / len * width;
    let py = dx / len * width;
    
    vec![
        Vertex { position: [x1 - px, y1 - py], color },
        Vertex { position: [x1 + px, y1 + py], color },
        Vertex { position: [x2 + px, y2 + py], color },
        
        Vertex { position: [x1 - px, y1 - py], color },
        Vertex { position: [x2 + px, y2 + py], color },
        Vertex { position: [x2 - px, y2 - py], color },
    ]
}
