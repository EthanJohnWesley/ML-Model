use csv::Reader;
use std::fs::File;
use ndarray::{ Array, Array1, Array };
use linfa::Dataset;

fn get_dataset() -> Dataset<f32, i32, ndarray::Dim<[usize; 1]>> {
    let mut reader = Reader::from_path("./src/heart").unwrap();

    let headers = get_headers(&mut reader);
    let data = get_data(&mut reader);
    let target_index = headers.len() - 1;

    let features = headers[0..target_index].to_vec();
    let records = get_records(&data, target_index);
    let targets = get_targets(&data, target_index);

    return Dataset::new(records, targets)
        .with_feature_names(features)
}
