1. FILE SIZE LIMITS
   Rule Limit
   Max lines per .rs file 300 lines
   Max lines per function 50 lines
   Max items exported from mod.rs 10 items
   If a file exceeds 300 lines → split into sub-modules.

2. MODULE STRUCTURE
   src/
   ├── lib.rs # ONLY: mod declarations + #[wasm_bindgen] init
   ├── bridge/
   │ ├── mod.rs # ONLY: pub use re-exports
   │ └── landmarks.rs # Single feature
   ├── physics/
   │ ├── mod.rs
   │ ├── kalman.rs # One struct per file
   │ ├── depth.rs
   │ ├── angles.rs
   │ └── velocity.rs
   ├── renderer/
   │ ├── mod.rs
   │ ├── skeleton.rs
   │ └── effects.rs
   └── game/
   ├── mod.rs
   ├── detection.rs
   └── scoring.rs
   Rules:
   mod.rs = Only re-exports, no logic
   One primary struct per file (KalmanFilter → kalman.rs)
   Group by domain (physics/, renderer/, game/)
3. NAMING CONVENTIONS
   Functions
   // ✅ GOOD - Descriptive, self-documenting
   fn calculate_anthropometric_depth(arm_length_2d: f32, calibrated_length: f32) -> f32
   // ❌ BAD - Cryptic, needs comments
   fn calc_d(l: f32, c: f32) -> f32
   Structs
   // ✅ GOOD
   pub struct KalmanStateEstimator { ... }
   pub struct ArmLandmarks { ... }
   // ❌ BAD
   pub struct KSE { ... }
   pub struct Data { ... }
   Constants
   // ✅ GOOD - Grouped, documented
   pub mod config {
   pub const VELOCITY_THRESHOLD: f32 = 0.17;
   pub const COOLDOWN_FRAMES: u32 = 15;
   pub const COMBO_TIMEOUT_FRAMES: u32 = 45;
   }
4. lib.rs TEMPLATE
   //! Boxing Web - WebGPU Punch Detection Game
   mod bridge;
   mod physics;
   mod renderer;
   mod game;
   use wasm_bindgen::prelude::\*; #[wasm_bindgen(start)]
   pub fn init() -> Result<(), JsValue> {
   console_error_panic_hook::set_once();
   // Initialize subsystems
   Ok(())
   }
   // ONLY wasm-bindgen entry points here
   // All logic lives in submodules
5. COMMENT STYLE
   // ❌ AVOID - Explaining "what"
   // Calculate the depth
   let depth = calculate_depth(arm);
   // ✅ PREFER - Explaining "why" (only when non-obvious)
   // Pythagorean theorem: Z = sqrt(L² - P²) where L=calibrated, P=projected
   let depth = (calibrated_length.powi(2) - projected_length.powi(2)).sqrt();
6. ERROR HANDLING
   // ✅ GOOD - Explicit error types
   pub enum PhysicsError {
   NotCalibrated,
   InvalidLandmarkData,
   }
   pub fn calculate_depth(...) -> Result<f32, PhysicsError>
   // ❌ BAD - Panic or unwrap in library code
   let depth = data.unwrap(); // Don't do this
7. WEB-SYS / JS INTEROP
   Keep all #[wasm_bindgen] functions in bridge/ module:

// bridge/landmarks.rs #[wasm_bindgen]
pub fn update_landmarks(data: &[f32]) {
// Validate, then delegate to internal systems
if let Ok(landmarks) = parse_landmarks(data) {
PHYSICS_ENGINE.lock().unwrap().update(landmarks);
}
}
// Internal API (no wasm_bindgen)
fn parse_landmarks(data: &[f32]) -> Result<Landmarks, Error> { ... } 8. CHECKLIST BEFORE EACH FILE
Under 300 lines?
Single responsibility?
Descriptive function names?
No logic in mod.rs? #[wasm_bindgen] only in bridge/?
Errors handled, not unwrapped?
