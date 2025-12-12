//! GPU State management - WebGPU device, queue, surface initialization

use std::cell::RefCell;
use wasm_bindgen::prelude::*;
use super::shapes::Vertex;

/// Errors that can occur during GPU initialization
pub enum GpuStateError {
    NoWindow,
    NoDocument,
    NoCanvas,
    SurfaceCreationFailed(String),
    NoAdapter,
    DeviceCreationFailed(String),
}

impl From<GpuStateError> for JsValue {
    fn from(err: GpuStateError) -> Self {
        match err {
            GpuStateError::NoWindow => JsValue::from_str("No window found"),
            GpuStateError::NoDocument => JsValue::from_str("No document found"),
            GpuStateError::NoCanvas => JsValue::from_str("No canvas with id 'game-canvas' found"),
            GpuStateError::SurfaceCreationFailed(e) => JsValue::from_str(&format!("Surface creation failed: {}", e)),
            GpuStateError::NoAdapter => JsValue::from_str("Failed to find a suitable GPU adapter"),
            GpuStateError::DeviceCreationFailed(e) => JsValue::from_str(&format!("Device creation failed: {}", e)),
        }
    }
}

/// Holds all WebGPU state for rendering
pub(crate) struct GpuState {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub render_pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
}

// Thread-local storage for GPU state (WASM is single-threaded)
thread_local! {
    pub(crate) static GPU_STATE: RefCell<Option<GpuState>> = RefCell::new(None);
}

/// Initialize WebGPU: adapter, device, surface, pipeline
pub async fn initialize_gpu() -> Result<(), GpuStateError> {
    let window = web_sys::window().ok_or(GpuStateError::NoWindow)?;
    let document = window.document().ok_or(GpuStateError::NoDocument)?;
    let canvas = document
        .get_element_by_id("game-canvas")
        .ok_or(GpuStateError::NoCanvas)?
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .map_err(|_| GpuStateError::NoCanvas)?;

    canvas.set_width(800);
    canvas.set_height(600);

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        ..Default::default()
    });

    let surface = instance
        .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
        .map_err(|e| GpuStateError::SurfaceCreationFailed(format!("{:?}", e)))?;

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .ok_or(GpuStateError::NoAdapter)?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Boxing Web Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        )
        .await
        .map_err(|e| GpuStateError::DeviceCreationFailed(format!("{:?}", e)))?;

    // Configure surface
    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = surface_caps
        .formats
        .iter()
        .find(|f| f.is_srgb())
        .copied()
        .unwrap_or(surface_caps.formats[0]);

    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: 800,
        height: 600,
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);

    // Create shader and pipeline
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Landmark Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shader.wgsl").into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Landmark Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[Vertex::desc()],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Vertex Buffer"),
        size: 4096 * std::mem::size_of::<Vertex>() as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    GPU_STATE.with(|state| {
        *state.borrow_mut() = Some(GpuState {
            device,
            queue,
            surface,
            render_pipeline,
            vertex_buffer,
        });
    });

    Ok(())
}
