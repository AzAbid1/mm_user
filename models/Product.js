const mongoose = require('mongoose');

const productSchema = new mongoose.Schema({
  name: { type: String, required: true },
  imageFileNames: [{ type: String }], // Changed to array for multiple images
  price: { type: Number, required: true },
  category: { type: String, required: true },
  user: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  description : { type: String, required: false }
}, { timestamps: true });

module.exports = mongoose.model('Product', productSchema);