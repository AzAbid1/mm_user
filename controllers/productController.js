const Product = require('../models/Product');
const fs = require('fs');
const path = require('path');

const createProduct = async (req, res) => {
  try {
    const { name, price, category, user } = req.body;

    // Validate required fields
    if (!req.file) {
      return res.status(400).json({ message: 'Image file is required' });
    }

    const imageFileName = req.file.filename;

    const newProduct = new Product({
      name,
      imageFileName,
      price,
      category,
      user
    });

    await newProduct.save();
    res.status(201).json({ message: 'Product created successfully', product: newProduct });
  } catch (error) {
    console.error('Error creating product:', error);
    res.status(500).json({ message: 'Server error', error: error.message });
  }
};




const updateProduct = async (req, res) => {
  try {
    const productId = req.params.id;
    const { name, price, category } = req.body;

    const updateFields = { name, price, category };

    const product = await Product.findById(productId);
    if (!product) {
      return res.status(404).json({ message: 'Product not found' });
    }

    // If a new file is uploaded, delete the old one and use the new filename
    if (req.file) {
      const newFileName = req.file.filename;

      // Delete old file from local storage if it exists
      if (product.imageFileName) {
        const oldFilePath = path.join(__dirname, '../public', product.imageFileName);
        if (fs.existsSync(oldFilePath)) {
          fs.unlinkSync(oldFilePath);
        }
      }

      updateFields.imageFileName = newFileName;
    }

    const updatedProduct = await Product.findByIdAndUpdate(
      productId,
      updateFields,
      { new: true, runValidators: true }
    );

    res.status(200).json({ message: 'Product updated successfully', product: updatedProduct });
  } catch (error) {
    console.error('Error updating product:', error);
    res.status(500).json({ message: 'Server error', error: error.message });
  }
};


const deleteProduct = async (req, res) => {
  try {
    const productId = req.params.id;

    const deletedProduct = await Product.findByIdAndDelete(productId);
    if (!deletedProduct) {
      return res.status(404).json({ message: 'Product not found' });
    }

    // Delete associated image from local file system
    if (deletedProduct.imageFileName) {
      const filePath = path.join(__dirname, '../public', deletedProduct.imageFileName);
      if (fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
      }
    }

    res.status(200).json({ message: 'Product deleted successfully' });
  } catch (error) {
    console.error('Error deleting product:', error);
    res.status(500).json({ message: 'Server error', error: error.message });
  }
};

const getProductById = async (req, res) => {
  try {
    const productId = req.params.id;

    const product = await Product.findById(productId);
    if (!product) {
      return res.status(404).json({ message: 'Product not found' });
    }

    let imageUrl = null;
    if (product.imageFileName) {
      const filePath = path.join(__dirname, '../public', product.imageFileName);
      if (fs.existsSync(filePath)) {
        imageUrl = `/images/${product.imageFileName}`; // relative to your domain
      }
    }

    res.status(200).json({
      product: { ...product.toObject(), imageUrl }
    });
  } catch (error) {
    console.error('Error fetching product:', error);
    res.status(500).json({ message: 'Server error', error: error.message });
  }
};



const getAllProducts = async (req, res) => {
  try {
    const userId = req.params.userId;

    const products = await Product.find({ user: userId });

    const productsWithUrls = products.map(product => {
      let imageUrl = null;
      const filePath = path.join(__dirname, '../public', product.imageFileName || '');
      if (product.imageFileName && fs.existsSync(filePath)) {
        imageUrl = `/images/${product.imageFileName}`;
      }
      return { ...product.toObject(), imageUrl };
    });

    res.status(200).json({ products: productsWithUrls });
  } catch (error) {
    console.error('Error fetching products:', error);
    res.status(500).json({ message: 'Server error', error: error.message });
  }
};




module.exports = {
  createProduct,
  updateProduct,
  deleteProduct,
  getProductById,
  getAllProducts
};