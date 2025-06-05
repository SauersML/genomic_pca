// build.rs
// This script configures the Rust compiler to optimize release builds
// specifically for the CPU of the machine performing the compilation.

use std::env;

fn main() {
    // Tell Cargo to only re-run this build script if build.rs itself changes.
    println!("cargo:rerun-if-changed=build.rs");

    // Get the current build profile. Cargo sets this environment variable.
    // We only want to apply aggressive optimizations for 'release' builds.
    let profile = env::var("PROFILE").unwrap_or_else(|_| {
        eprintln!("[build.rs] WARN: PROFILE environment variable not found. Assuming 'debug' build. No specific SIMD flags will be applied by this script.");
        "debug".to_string()
    });

    if profile == "release" {
        // FOR RELEASE BUILDS:
        // Enable "-C target-cpu=native".
        // This instructs `rustc` to detect all features of the current build host's CPU
        // (including all available SIMD instruction sets like SSE, AVX, AVX2, AVX512 on x86-64,
        // or NEON and its extensions on AArch64) and optimize the output binary specifically for it.
        
        println!("cargo:rustc-flags=-C target-cpu=native");

        eprintln!("[build.rs] Configuring for RELEASE build: Applying '-C target-cpu=native'.");
    } else {
        // For DEBUG builds (or any profile other than 'release'):
        eprintln!("[build.rs] Profile: '{}'. No specific SIMD optimization flags applied by this script. Compiler defaults and `std::simd` scalar fallbacks will be used.", profile);
    }
}
