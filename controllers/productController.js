const Product = require('../models/Product');
const fs = require('fs');
const path = require('path');

const createProduct = async (req, res) => {
  try {
    const { name, price, category, user, description } = req.body;  // Add description

    // Validate required fields
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ message: 'At least one image file is required' });
    }

    // Limit to 5 images
    if (req.files.length > 5) {
      return res.status(400).json({ message: 'Maximum 5 images allowed' });
    }

    const imageFileNames = req.files.map(file => file.filename);

    const newProduct = new Product({
      name,
      imageFileNames,
      price,
      category,
      user,
      description  // Add description
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
    const { name, price, category, deleteImages, description } = req.body;  // Add description
    const updateFields = { name, price, category, description };  // Add description
     
    const product = await Product.findById(productId);
    if (!product) {
      return res.status(404).json({ message: 'Product not found' });
    }
    console.log("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    console.log(description);
    // Parse deleteImages (may be JSON string or array)
    let imagesToDelete = [];
    try {
      imagesToDelete = typeof deleteImages === 'string' ? JSON.parse(deleteImages) : deleteImages || [];
    } catch (error) {
      console.error('Error parsing deleteImages:', error);
    }

    // Initialize new imageFileNames with existing ones
    let newImageFileNames = product.imageFileNames ? [...product.imageFileNames] : [];

    // Remove specified images
    if (imagesToDelete.length > 0) {
      newImageFileNames = newImageFileNames.filter(filename => !imagesToDelete.includes(filename));
      // Delete files from public folder
      imagesToDelete.forEach(fileName => {
        const filePath = path.join(__dirname, '../public', fileName);
        if (fs.existsSync(filePath)) {
          fs.unlinkSync(filePath);
        }
      });
    }

    // Handle new images
    if (req.files && req.files.length > 0) {
      const newFiles = req.files.map(file => file.filename);
      newImageFileNames = [...newImageFileNames, ...newFiles];
    }

    // Enforce 5-image limit
    if (newImageFileNames.length > 5) {
      // Clean up any newly uploaded files if limit is exceeded
      if (req.files && req.files.length > 0) {
        req.files.forEach(file => {
          const filePath = path.join(__dirname, '../public', file.filename);
          if (fs.existsSync(filePath)) {
            fs.unlinkSync(filePath);
          }
        });
      }
      return res.status(400).json({ message: 'Maximum 5 images allowed' });
    }

    updateFields.imageFileNames = newImageFileNames;

    const updatedProduct = await Product.findByIdAndUpdate(
      productId,
      updateFields,
      { new: true, runValidators: true }
    );

    res.status(200).json({ message: 'Product updated successfully', product: updatedProduct });
  } catch (error) {
    console.error('Error updating product:', error);
    // Clean up any uploaded files on error
    if (req.files && req.files.length > 0) {
      req.files.forEach(file => {
        const filePath = path.join(__dirname, '../public', file.filename);
        if (fs.existsSync(filePath)) {
          fs.unlinkSync(filePath);
        }
      });
    }
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

    // Delete associated images from local file system
    if (deletedProduct.imageFileNames && deletedProduct.imageFileNames.length > 0) {
      deletedProduct.imageFileNames.forEach(fileName => {
        const filePath = path.join(__dirname, '../public', fileName);
        if (fs.existsSync(filePath)) {
          fs.unlinkSync(filePath);
        }
      });
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

    const imageUrls = product.imageFileNames.map(fileName => {
      const filePath = path.join(__dirname, '../public', fileName);
      return fs.existsSync(filePath) ? `/images/${fileName}` : null;
    }).filter(url => url !== null);

    res.status(200).json({
      product: { ...product.toObject(), imageUrls }
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
      const imageUrls = product.imageFileNames.map(fileName => {
        const filePath = path.join(__dirname, '../public', fileName);
        return fs.existsSync(filePath) ? `/images/${fileName}` : null;
      }).filter(url => url !== null);
      return { ...product.toObject(), imageUrls };
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