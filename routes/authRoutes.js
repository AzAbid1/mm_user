const express = require('express');
const passport = require('passport');
const { registerUser, loginUser, googleLoginCallback } = require('../controllers/authController');

const router = express.Router();

// Route for account creation (Registration)
router.post('/register', registerUser);

// Route for user login
router.post('/login', loginUser);

// Google OAuth Route
router.get('/auth/google', passport.authenticate('google', { scope: ['profile', 'email'] }));

// Google OAuth callback route
router.get('/auth/google/callback', passport.authenticate('google', { session: false }), googleLoginCallback);

module.exports = router;
