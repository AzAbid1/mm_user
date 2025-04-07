const express = require('express');
const passport = require('passport');
const { 
    registerUser, 
    loginUser, 
    googleLoginCallback,
    getUserById,
    updateUserDetails,
    changePassword
 } = require('../controllers/authController');

const router = express.Router();

// Route for account creation (Registration)
router.post('/register', registerUser);

// Route for user login
router.post('/login', loginUser);

// Google OAuth Route
router.get('/auth/google', passport.authenticate('google', { scope: ['profile', 'email'] }));

// Google OAuth callback route
router.get('/auth/google/callback', passport.authenticate('google', { session: false }), googleLoginCallback);

router.get('/user/:id', getUserById);

// Update user details
router.put('/user/:id', updateUserDetails);

// Change password
router.put('/user/:id/change-password', changePassword);

module.exports = router;
