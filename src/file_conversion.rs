#[cfg(feature = "mtx")]
use std::ops::Index;
#[cfg(feature = "mtx")]
use matrix_merchant::{
    reader::*, 
    writer::MatrixWriter,
    MatrixSize,
    Position,
};
#[cfg(feature = "mtx")]
use num_complex::Complex;

#[cfg(feature = "hdf5")]
use hdf5::Dataset;

use std::{
    error::Error,
    fs,
    io,
    hash::{Hash, Hasher, DefaultHasher},
};
use crate::vector::vector_builders::VectorBuilder;
use std::path::{Path, PathBuf};


use crate::vector::{
    VectorOps,
    vec_util_traits::Get, 
    vector_exprs::UninitVectorExpr, 
    vector_exprs::ConcreteVectorExpr
};

pub trait AsText: Sized {
    type Error: Error;

    fn as_text(&self) -> String;
    fn as_text_utf8_bytes(&self) -> Vec<u8> {
        self.as_text().into_bytes()
    }
    fn from_text(text: &str) -> Result<Self, Self::Error>;
}

pub trait AsData<E: Error>: Sized {
    type Error: Error;

    fn as_data(&self) -> Vec<u8>;
    fn from_data(data: &[u8]) -> Result<Self, Self::Error>;
}

#[derive(Debug)]
#[cfg(feature = "csv")]
pub enum CSVError<FieldError: Error> {
    CSVError(csv::Error),
    CellOutOfBounds,
    FieldError(FieldError),
    FileError(io::Error),
}

impl<FieldError: Error> std::fmt::Display for CSVError<FieldError> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CSVError(e) => write!(f, "{}", e),
            Self::CellOutOfBounds => write!(f, "CellOutOfBounds"),
            Self::FieldError(e) => write!(f, "{}", e),
            Self::FileError(e) => write!(f, "{}", e),
        }
    }
}

impl<FieldError: Error> From<csv::Error> for CSVError<FieldError> {
    fn from(value: csv::Error) -> Self {
        Self::CSVError(value)
    }
}

impl<FieldError: Error> From<io::Error> for CSVError<FieldError> {
    fn from(value: io::Error) -> Self {
        Self::FileError(value)
    }
}

impl<FieldError: Error> Error for CSVError<FieldError> {}

#[derive(Debug)]
#[cfg(feature = "mtx")]
pub enum MTXError {
    NonVector,
    WrongType,
    MTXError(matrix_merchant::Error),
    FileError(io::Error),
}

impl From<matrix_merchant::Error> for MTXError {
    fn from(value: matrix_merchant::Error) -> Self {
        MTXError::MTXError(value)
    }
}

impl From<io::Error> for MTXError {
    fn from(value: io::Error) -> Self {
        MTXError::FileError(value)
    }
}

fn generate_tmp_file_path<P: AsRef<Path>>(path: P) -> PathBuf {
    let mut temp_file = std::env::temp_dir();
    let mut hasher = DefaultHasher::new();
    path.as_ref().hash(&mut hasher);
    temp_file.push(format!("MathVectorTemp{:016X}", hasher.finish()));
    temp_file
}

struct IterInsert<I: Iterator> where I::Item: Clone {
    idx: usize,
    iter: I,
    insertion_item: Option<I::Item>,
    insertion_idx: usize,
    filler: I::Item,
}

impl<I: Iterator> IterInsert<I> where I::Item: Clone {
    fn new(iter: I, item: I::Item, idx: usize, filler: I::Item) -> Self {
        Self {
            idx: 0,
            iter,
            insertion_item: Some(item),
            insertion_idx: idx,
            filler,
        }
    }
}

impl<I: Iterator> Iterator for IterInsert<I> where I::Item: Clone {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let normal_item = self.iter.next();
        let out = if normal_item.is_none() & (self.idx < self.insertion_idx) {
            Some(self.filler.clone())
        } else if self.idx == self.insertion_idx {
            self.insertion_item.take()
        } else {
            normal_item
        };
        self.idx += 1;
        out
    }
}

#[derive(Debug)]
#[cfg(feature = "hdf5")]
pub enum HDF5Error {
    WrongSize,
    HDF5Error(hdf5::Error)
}

impl From<hdf5::Error> for HDF5Error {
    fn from(value: hdf5::Error) -> Self {
        Self::HDF5Error(value)
    }
}

#[cfg(feature = "mtx")]
macro_rules! mtx_read_fns {
    ($($read_fn_name:ident $field_name:ident $ty:ty;)*) => {
        $(
            fn $read_fn_name<P: AsRef<Path>>(path: P, builder: Self::Builder) -> Result<Self, MTXError> where Self: Index<usize, Output = $ty> {
                let reader = MtxReader::new_reader(fs::File::open(path)?)?;
                match reader.matrix().unwrap() {
                    MatrixReader::MatrixArray(array_reader) => {
                        let mut uninit = Self::new_uninit(builder);
                        let mut num_fields_written = 0;
                        let mut error = None;
                    
                        let MatrixSize {num_rows, num_cols} = array_reader.size();
                        if (num_rows != 1) & (num_cols != 1) {
                            error = Some(MTXError::NonVector);
                        } 
                    
                        let MatrixArrayReader::$field_name(array_reader) = array_reader else {return Err(MTXError::WrongType)};
                        for column in array_reader {
                            let column = match column {
                                Ok(column) => column,
                                Err(e) => {error = Some(e.into()); break}
                            };
                            for field in column {
                                unsafe { 
                                    Self::init_index(&mut uninit, num_fields_written, field);
                                    num_fields_written += 1;
                                }
                            }
                        }
                    
                        if let Some(error) = error {
                            unsafe {
                                for i in 0..num_fields_written {
                                    Self::drop_index(&mut uninit, i);
                                }
                                Self::drop_ots(&mut uninit);
                            }
                            return Err(error);
                        }
                        unsafe { Ok(Self::assume_init(uninit)) }
                    }
                    MatrixReader::MatrixCoordinate(coord_reader) => {
                        use matrix_merchant::Position;
                    
                        let MatrixSize {num_rows, num_cols} = coord_reader.size();
                        if (num_rows != 1) & (num_cols != 1) {
                            return Err(MTXError::NonVector);
                        }
                    
                        let mut vector = Self::new_zeroed(builder);
                        let MatrixCoordinateReader::$field_name(coord_reader) = coord_reader else {return Err(MTXError::WrongType)};
                        for field_data in coord_reader {
                            let (Position { row, col }, field) = field_data?;
                            if num_rows == 1 {
                                vector[col] = field;
                            } else {
                                vector[row] = field;
                            }
                        }
                    
                        Ok(vector)
                    }
                }
            }
        )*
    };
}

pub trait VectorFileConversions: UninitVectorExpr + ConcreteVectorExpr 
where 
    Self::Unwrapped: Get<Item = Self::Output>,
    Self::Output: Sized,
{
    #[cfg(feature = "csv")]
    /// top left corner == (row = 0, col = 0), vector is read top down
    fn read_csv_column<P: AsRef<Path>>(path: P, builder: Self::Builder, row_start: usize, col: usize) -> Result<Self, CSVError<<Self::Output as AsText>::Error>> where Self::Output: AsText {
        let mut csv = csv::Reader::from_path(path)?;
        let mut records = csv.records();
        for _ in 0..row_start {let _ = records.next().ok_or(CSVError::CellOutOfBounds)?;} // don't care if these individual rows are malformed
        let mut uninit = Self::new_uninit(builder);
        let mut num_fields_written = 0;
        let mut error = None;

        // Note: this section *must not return/crash* to avoid leaking
        for _ in 0..uninit.size() {
            let record = match records.next() {
                None => {
                    error = Some(CSVError::CellOutOfBounds);
                    break;
                }
                Some(Err(e)) => {
                    error = Some(CSVError::CSVError(e));
                    break;
                }
                Some(Ok(record)) => record
            };
            if col >= record.len() {
                error = Some(CSVError::CellOutOfBounds);
                break;
            }
            let field = match Self::Output::from_text(&record[col]) {
                Err(e) => {
                    error = Some(CSVError::FieldError(e));
                    break;
                }
                Ok(field) => field
            };
            unsafe { Self::init_index(&mut uninit, num_fields_written, field); }
            num_fields_written += 1;
        }

        if let Some(error) = error {
            unsafe {
                for i in 0..num_fields_written {
                    Self::drop_index(&mut uninit, i);
                }
                Self::drop_ots(&mut uninit);
            }
            return Err(error);
        }
        unsafe { Ok(Self::assume_init(uninit)) }
    }

    #[cfg(feature = "csv")]
    fn read_csv_row<P: AsRef<Path>>(path: P, builder: Self::Builder, row: usize, col_start: usize) -> Result<Self, CSVError<<Self::Output as AsText>::Error>> where Self::Output: AsText {
        let mut csv = csv::Reader::from_path(path)?;
        let mut records = csv.records();
        for _ in 0..row {let _ = records.next().ok_or(CSVError::CellOutOfBounds)?;} // don't care if these individual rows are malformed
        let mut uninit = Self::new_uninit(builder);
        let mut num_fields_written = 0;
        let mut error = None;

        // Note: this section *must not return/crash* to avoid leaking
        match records.next() {
            None => {
                error = Some(CSVError::CellOutOfBounds);
            }
            Some(Err(e)) => {
                error = Some(CSVError::CSVError(e));
            }
            Some(Ok(record)) => {
                for field in record.into_iter().skip(col_start).take(uninit.size()) {
                    let field = match Self::Output::from_text(field) {
                        Err(e) => {
                            error = Some(CSVError::FieldError(e));
                            break;
                        }
                        Ok(field) => field
                    };
                    unsafe {
                        Self::init_index(&mut uninit, num_fields_written, field);
                        num_fields_written += 1;
                    }
                }
            }
        };

        if let Some(error) = error {
            unsafe {
                for i in 0..num_fields_written {
                    Self::drop_index(&mut uninit, i);
                }
                Self::drop_ots(&mut uninit);
            }
            return Err(error);
        }
        unsafe { Ok(Self::assume_init(uninit)) }
    }

    #[cfg(feature = "csv")]
    fn write_csv_column<P: AsRef<Path>>(&self, path: P, row_start: usize, col: usize) -> Result<(), CSVError<<Self::Output as AsText>::Error>> 
    where 
        Self::Output: AsText,
    {
        let temp_path = generate_tmp_file_path(&path);
        let mut writer = csv::Writer::from_path(&temp_path)?;
        let mut base_records = if path.as_ref().exists() {
            Some(csv::Reader::from_path(&path)?)
        } else {
            None
        };

        let mut record = csv::StringRecord::new();
        for _ in 0..row_start {
            if let Some(base_records) = &mut base_records {
                base_records.read_record(&mut record)?; // Note: not sure if I need to clear or not
            }
            writer.write_record(&record)?;
        }

        for i in 0..self.size() {
            if let Some(base_records) = &mut base_records {
                base_records.read_record(&mut record)?;// Note: not sure if I need to clear or not
            }
            let field_txt = self[i].as_text();
            writer.write_record(IterInsert::new(record.iter(), &field_txt, col, ""))?;
        }

        if let Some(base_records) = &mut base_records {
            while base_records.read_record(&mut record)? { // Note: not sure if I need to clear or not
                writer.write_record(&record)?;
            }
        }

        writer.flush()?;
        // to make sure the files get closed
        drop(writer);
        drop(base_records);
        
        fs::rename(temp_path, path)?;
        Ok(())
    }

    #[cfg(feature = "csv")]
    fn write_csv_row<P: AsRef<Path>>(&self, path: P, row: usize, col_start: usize) -> Result<(), CSVError<<Self::Output as AsText>::Error>> 
    where 
        Self::Output: AsText,
    {
        let temp_path = generate_tmp_file_path(&path);
        let mut writer = csv::Writer::from_path(&temp_path)?;
        let mut base_records = if path.as_ref().exists() {
            Some(csv::Reader::from_path(&path)?)
        } else {
            None
        };

        let mut record = csv::StringRecord::new();
        for _ in 0..row {
            if let Some(base_records) = &mut base_records {
                base_records.read_record(&mut record)?; // Note: not sure if I need to clear or not
            }
            writer.write_record(&record)?;
        }

        if let Some(base_records) = &mut base_records {
            base_records.read_record(&mut record)?; // Note: not sure if I need to clear or not
        }
        let mut new_record = csv::StringRecord::new();
        let mut old_record = record.iter();
        for _ in 0..col_start {
            new_record.push_field(old_record.next().or(Some("")).unwrap())
        }
        for i in 0..self.size() {
            new_record.push_field(&self[i].as_text());
        }
        new_record.extend(old_record.skip(self.size()));

        if let Some(base_records) = &mut base_records {
            while base_records.read_record(&mut record)? { // Note: not sure if I need to clear or not
                writer.write_record(&record)?;
            }
        }

        writer.flush()?;
        // to make sure the files get closed
        drop(writer);
        drop(base_records);
        
        fs::rename(temp_path, path)?;
        todo!()
    }

    #[cfg(feature = "mtx")]
    mtx_read_fns!(
        read_mtx_real Real f64;
        read_mtx_integer Integer i64;
        read_mtx_complex Complex Complex<f64>;
    );

    #[cfg(feature = "mtx")]
    fn write_mtx<P: AsRef<Path>, C: AsRef<str>>(&self, path: P, comment: Option<C>) -> Result<(), MTXError> where Self::Output: matrix_merchant::Field {
        let temp_path = generate_tmp_file_path(&path);
        let mut writer = MatrixWriter::new(fs::File::open(&temp_path)?, self.size(), 1);
        if let Some(comment) = comment {
            writer.add_comment(comment)?;
        }
        writer.write_array(|Position {row, col: _}| &self[row])?;

        fs::rename(temp_path, path)?;
        Ok(())
    }

    #[cfg(feature = "hdf5")]
    fn read_hdf5_dataset(builder: Self::Builder, dataset: &Dataset) -> Result<Self, HDF5Error> where Self::Output: hdf5::H5Type {
        let data = dataset.read_1d()?;
        if data.len() != builder.size() {
            return Err(HDF5Error::WrongSize)
        }
        let mut uninit = Self::new_uninit(builder);
        let mut num_written = 0;
        for field in data {
            unsafe {
                Self::init_index(&mut uninit, num_written, field);
                num_written += 1;
            }
        }
        // assuming that this ^^^ is infallible
        
        Ok(unsafe { Self::assume_init(uninit) })
    }

    #[cfg(feature = "hdf5")]
    fn write_hdf5_dataset(&self, dataset: &Dataset) -> Result<(), HDF5Error> where Self::Output: hdf5::H5Type + Clone + ndarray::DataOwned {
        // note: if rust-analyzer says that this is an error, its lying 
        Ok(dataset.as_writer().write((0..self.size()).into_iter().map(|x| self[x].clone()).collect::<Vec<_>>())?)
    }
}

impl<V: UninitVectorExpr + ConcreteVectorExpr> VectorFileConversions for V 
where 
    Self::Unwrapped: Get<Item = Self::Output>,
    Self::Output: Sized,
{}