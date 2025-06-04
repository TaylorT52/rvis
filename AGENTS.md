# Agents

This project runs on Rust 1.89.0-nightly (2025-05-31). Feel free to use any nightly features you'd like.

This project is split into two crates:
- `net`: Contains the user level test code. Do not modify this crate.
- `tensor`: Contains the tensor library. This should be modified and where features are added.

The goal of this project is to create a tensor library which is 1) portable across different backends and 2) uses 
const generics for compile time verification of tensor shapes and operations. The idea is that a user should never run
into runtime errors due to shape mismatches or incorrect tensor operations.

