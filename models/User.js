const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
    email: { type: String, required: true, unique: true },
    firstName: { type: String, required: true },
    lastName: { type: String, required: true },
    password: { type: String, default: "" }, // Empty for social logins
    isSocial: { type: Boolean, default: false } // True for Google login
});

const User = mongoose.model('User', userSchema);

module.exports = User;
