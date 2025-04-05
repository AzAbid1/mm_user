require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const passport = require('passport');
const app = express();
const port = process.env.PORT || 3000;
const cors = require("cors");
require('./config/passportConfig');

// Middleware to parse JSON requests
app.use(express.json());

// CORS
const corsOptions = {
    origin: function (origin, callback) {
      const allowedOrigins = ['http://localhost:4200', 'http://localhost:3000'];
      if (allowedOrigins.indexOf(origin) !== -1) {
        callback(null, true);
      } else {
        callback(null, false);
      }
    },
    credentials: true
  };
  app.use(cors(corsOptions));

app.use(passport.initialize());

// Import routes
const authRoutes = require('./routes/authRoutes');

// Connect to MongoDB
mongoose.connect(process.env.MONGO_URI)
    .then(() => console.log('✅ MongoDB connected successfully'))
    .catch(err => console.error('❌ MongoDB connection error:', err));

// Use routes
app.use(authRoutes);

// Default route
app.get('/', (req, res) => {
    res.send('Hello, Express!');
});

// Start the server
app.listen(port, () => {
    console.log(`🚀 Server running at http://localhost:${port}`);
});
