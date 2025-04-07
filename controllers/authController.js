const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const User = require('../models/User');
const JWT_SECRET = process.env.JWT_SECRET;

// Register route controller
const registerUser = async (req, res) => {
    try {
        const { email, firstName, lastName, password } = req.body;

        // Check if user exists
        const existingUser = await User.findOne({ email });
        if (existingUser) return res.status(400).json({ message: "User already exists" });

        // Hash password
        const hashedPassword = await bcrypt.hash(password, 10);

        // Create user
        const newUser = new User({ email, firstName, lastName, password: hashedPassword });
        await newUser.save();

        // Generate JWT token
        const token = jwt.sign(
            { id: newUser._id },
            JWT_SECRET, 
            { expiresIn: 2592000 }
        );

        res.status(201).json({
            message: "Account created successfully",
            token,
            user: newUser
        });
    } catch (error) {
        res.status(500).json({ message: "Server error", error: error.message });
    }
};

// Login route controller
const loginUser = async (req, res) => {
    try {
        const { email, password } = req.body;

        const user = await User.findOne({ email });
        if (!user) return res.status(400).json({ message: "Invalid email or password" });

        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch) return res.status(400).json({ message: "Invalid email or password" });

        const token = jwt.sign(
            { id: user._id },
            JWT_SECRET, 
            { expiresIn: 2592000 });

        res.status(200).json({ message: "Login successful", token, user });
    } catch (error) {
        res.status(500).json({ message: "Server error", error: error.message });
    }
};

// Google login callback controller
const googleLoginCallback = (req, res) => {
    const token = req.user.token;
    const user = req.user.user;
    const redirectUrl = `${process.env.FRONTEND_URL}/auth/google-success?token=${token}&id=${user._id}`;
    res.redirect(redirectUrl);
};

// Get user by ID
const getUserById = async (req, res) => {
    try {
        const userId = req.params.id;

        const user = await User.findById(userId).select('-password'); // Exclude password
        if (!user) return res.status(404).json({ message: "User not found" });

        res.status(200).json({ user });
    } catch (error) {
        res.status(500).json({ message: "Server error", error: error.message });
    }
};

// Update account details
const updateUserDetails = async (req, res) => {
    try {
        const userId = req.params.id;
        const { firstName, lastName } = req.body;

        const updatedUser = await User.findByIdAndUpdate(
            userId,
            { firstName, lastName },
            { new: true, runValidators: true }
        ).select('-password'); // Don't return the password

        if (!updatedUser) return res.status(404).json({ message: "User not found" });

        res.status(200).json({ message: "User updated successfully", user: updatedUser });
    } catch (error) {
        res.status(500).json({ message: "Server error", error: error.message });
    }
};

// Change password
const changePassword = async (req, res) => {
    try {
        const userId = req.params.id;
        const { oldPassword, newPassword } = req.body;

        const user = await User.findById(userId);
        if (!user) return res.status(404).json({ message: "User not found" });

        const isMatch = await bcrypt.compare(oldPassword, user.password);
        if (!isMatch) return res.status(400).json({ message: "Old password is incorrect" });

        const hashedNewPassword = await bcrypt.hash(newPassword, 10);
        user.password = hashedNewPassword;

        await user.save();

        res.status(200).json({ message: "Password updated successfully" });
    } catch (error) {
        res.status(500).json({ message: "Server error", error: error.message });
    }
};


module.exports = {
    registerUser,
    loginUser,
    googleLoginCallback,
    getUserById,
    updateUserDetails,
    changePassword
};
