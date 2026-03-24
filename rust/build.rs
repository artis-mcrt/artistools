use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=../.git/HEAD");
    println!("cargo:warning=Running set_version.py");
    Command::new("python3").args(["../set_version.py"]);
}
