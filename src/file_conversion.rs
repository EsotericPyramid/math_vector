use std::{
    error::Error, 
    io::{Write, BufWriter, self},
    fs::File,
};

use crate::vector::{
    vec_util_traits::{Get, HasReuseBuf}, vector_initialization::UninitVectorExpr, vector_math::ConcreteVectorExpr, VectorOps
};

fn sanitize_csv_entry(entry: &str) -> String {
    // basically: put it in double quotes and double each double quote within
    let mut sanitized = String::with_capacity(entry.len() + 2);
    sanitized.push('"');
    for char in entry.chars() {
        sanitized.push(char);
        if char == '"' {
            sanitized.push('"');
        }
    }
    sanitized.push('"');
    sanitized
}

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

pub trait VectorFileConversions: UninitVectorExpr + ConcreteVectorExpr 
where 
    Self::Unwrapped: Get<Item = Self::Output>,
    Self::Output: Sized,
{
    /// inserts data corresponding to a CSV file of the vector formatted as a column
    /// *WARNING* does not intelligently format itself into a prexisting csv file, it just dumps the data whether or not thats correct
    /// *WARNING* does not sanitize the entries being written (ex: if they contain a comma or double quotes)
    fn raw_unsanitized_write_csv_column<'a, E>(&'a self, file: &mut File) -> Result<(), io::Error> 
    where 
        Self::ReferencedInner<'a>: HasReuseBuf<BoundTypes = <Self::ReferencedInner<'a> as Get>::BoundItems>,
        &'a Self::Output: AsText,
    {
        let mut file = BufWriter::new(file);
        for item in self.borrow().into_vec_iter() {
            file.write_all(&item.as_text_utf8_bytes())?;
            file.write_all(&"\n".as_bytes())?;
        }
        Ok(())
    }

    /// inserts data corresponding to a CSV file of the vector formatted as a column
    /// *WARNING* does not intelligently format itself into a prexisting csv file, it just dumps the data whether or not thats correct
    fn raw_write_csv_column<'a, E>(&'a self, file: &mut File) -> Result<(), io::Error> 
    where 
        Self::ReferencedInner<'a>: HasReuseBuf<BoundTypes = <Self::ReferencedInner<'a> as Get>::BoundItems>,
        &'a Self::Output: AsText,
    {
        let mut file = BufWriter::new(file);
        for item in self.borrow().into_vec_iter() {
            file.write_all(&sanitize_csv_entry(&item.as_text()).as_bytes())?;
            file.write_all(&"\n".as_bytes())?;
        }
        Ok(())
    }

    /// inserts data corresponding to a CSV file of the vector formatted as a row
    /// *WARNING* does not intelligently format itself into a prexisting csv file, it just dumps the data whether or not thats correct
    /// *WARNING* does not sanitize the entries being written (ex: if they contain a comma or double quotes)
    fn raw_unsanitized_write_csv_row<'a, E>(&'a self, file: &mut File) -> Result<(), io::Error> 
    where 
        Self::ReferencedInner<'a>: HasReuseBuf<BoundTypes = <Self::ReferencedInner<'a> as Get>::BoundItems>,
        &'a Self::Output: AsText,
    {
        let mut file = BufWriter::new(file);
        let mut iter = self.borrow().into_vec_iter();
        if let Some(item) = iter.next() {
            file.write_all(&item.as_text_utf8_bytes())?;
        }
        for item in iter {
            file.write_all(&",".as_bytes())?;
            file.write_all(&item.as_text_utf8_bytes())?;
        }
        Ok(())
    }

    /// inserts data corresponding to a CSV file of the vector formatted as a column
    /// *WARNING* does not intelligently format itself into a prexisting csv file, it just dumps the data whether or not thats correct
    fn raw_write_csv_row<'a, E>(&'a self, file: &mut File) -> Result<(), io::Error> 
    where 
        Self::ReferencedInner<'a>: HasReuseBuf<BoundTypes = <Self::ReferencedInner<'a> as Get>::BoundItems>,
        &'a Self::Output: AsText,
    {
        let mut file = BufWriter::new(file);
        let mut iter = self.borrow().into_vec_iter();
        if let Some(item) = iter.next() {
            file.write_all(&sanitize_csv_entry(&item.as_text()).as_bytes())?;
        }
        for item in iter {
            file.write_all(&",".as_bytes())?;
            file.write_all(&sanitize_csv_entry(&item.as_text()).as_bytes())?;
        }
        Ok(())
    }
}