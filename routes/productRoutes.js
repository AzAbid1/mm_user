const express = require('express');
const upload = require('../middleware/upload');
const {
  createProduct,
  updateProduct,
  deleteProduct,
  getProductById,
  getAllProducts
} = require('../controllers/productController');

const router = express.Router();

router.post('/products', upload.single('image'), createProduct);
router.put('/products/:id', upload.single('image'), updateProduct);
router.delete('/products/:id', deleteProduct);
router.get('/products/getOne/:id', getProductById);
router.get('/products/getAll/:userId', getAllProducts);

module.exports = router;
