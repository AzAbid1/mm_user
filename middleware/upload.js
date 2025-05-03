const multer = require('multer');
const path = require('path');
const crypto = require('crypto');

// Define storage config
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, path.join(__dirname, '../public'));
  },
  filename: function (req, file, cb) {
    const ext = path.extname(file.originalname);
    const uniqueName = crypto.randomBytes(16).toString('hex') + ext;
    cb(null, uniqueName);
  }
});

// Filter allowed file types (optional)
const fileFilter = (req, file, cb) => {
  if (file.mimetype.startsWith('image/')) cb(null, true);
  else cb(new Error('Only image files are allowed!'), false);
};

const upload = multer({ storage, fileFilter });

module.exports = upload;
