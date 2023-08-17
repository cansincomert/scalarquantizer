use std::io::Read;
use std::error::Error;
use std::fs::File;
use std::io::{self, BufReader};
use std::path::Path;
// use rand::distributions::{Distribution, Uniform};
use ScalarQuantizer::scalar_quantizer::ScalarQuantizer;


fn load_vectors_from_file<P: AsRef<Path>>(path: P, num_vectors: usize, dim: usize) -> io::Result<Vec<f32>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut buffer = Vec::with_capacity(dim * num_vectors);

    for _ in 0..num_vectors {
        // Read the dimension (though in standard .fvecs it should be equal to dim)
        let mut dim_buffer = [0u8; 4];
        reader.read_exact(&mut dim_buffer)?;
        let vec_dim = i32::from_le_bytes(dim_buffer) as usize;

        if vec_dim != dim {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Unexpected vector dimension."));
        }

        // Read the float values
        let mut vec_data = vec![0f32; vec_dim];
        let byte_data = unsafe {
            std::slice::from_raw_parts_mut(vec_data.as_mut_ptr() as *mut u8, 4 * vec_dim)
        };
        reader.read_exact(byte_data)?;
        buffer.extend_from_slice(&vec_data);
    }
    Ok(buffer)
}

use rand::seq::SliceRandom;  // Add this import for shuffling
use std::io::{Seek, SeekFrom};
use rand::Rng;

fn load_random_vectors_from_file<P: AsRef<Path>>(path: P, num_random_vectors: usize, dim: usize) -> io::Result<Vec<f32>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let vec_byte_size = 4 + 4 * dim;  // Byte size of each vector
    let total_vectors = reader.seek(SeekFrom::End(0))? / vec_byte_size as u64;  // Total number of vectors in the file

    let mut rng = rand::thread_rng();
    let mut random_indices: Vec<u64> = (0..total_vectors).collect();
    random_indices.shuffle(&mut rng);
    random_indices.truncate(num_random_vectors);

    let mut buffer = Vec::with_capacity(dim * num_random_vectors);

    for index in random_indices {
        reader.seek(SeekFrom::Start(index * vec_byte_size as u64))?;

        // Read the dimension
        let mut dim_buffer = [0u8; 4];
        reader.read_exact(&mut dim_buffer)?;
        let vec_dim = i32::from_le_bytes(dim_buffer) as usize;

        if vec_dim != dim {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Unexpected vector dimension."));
        }

        // Read the float values
        let mut vec_data = vec![0f32; vec_dim];
        let byte_data = unsafe {
            std::slice::from_raw_parts_mut(vec_data.as_mut_ptr() as *mut u8, 4 * vec_dim)
        };
        reader.read_exact(byte_data)?;

        buffer.extend_from_slice(&vec_data);
    }
    Ok(buffer)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Define the parameters
    let quantile = 0.99;
    let num_vectors = 10;  // Number of random vectors to read
    let dim = 960;

    // Load random vectors from the file (assuming gist_base.fvecs is in the src directory)
    let vecs = load_random_vectors_from_file("src/gist_base.fvecs", num_vectors, dim)?;  // Notice the function name change

    // Create a ScalarQuantizer instance
    let quantizer = ScalarQuantizer::new(quantile).expect("unable to create the quantizer");

    // Quantize the vectors
    let quantized_vecs = quantizer.quantize_arr(&vecs);

    // Display quantized vectors and sum
    println!("Quantized Vectors: {:?}", quantized_vecs);
    println!("Number of quantized dimensions: {}", quantized_vecs.0.len());

    Ok(())
}


// fn main() -> Result<(), Box<dyn Error>> {
//     // Define the parameters
//     let quantile = 0.99;
//     let num_vectors = 100;
//     let dim = 960;
//
//     // Load vectors from the file (assuming gist_base.ivecs is in the src directory)
//     let vecs = load_vectors_from_file("src/gist_base.fvecs", num_vectors, dim)?;
//
//     // Create a ScalarQuantizer instance
//     let quantizer = ScalarQuantizer::new(quantile).expect("unable to create the quantizer");
//
//     // Quantize the vectors
//     let quantized_vecs = quantizer.quantize_arr(&vecs);
//
//     // Display quantized vectors and sum
//     println!("Quantized Vectors: {:?}", quantized_vecs);
//     println!("{}", quantized_vecs.0.len());
//     Ok(())
// }
