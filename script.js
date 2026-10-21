// IVF Journey Tracker - Main JavaScript File

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

/**
 * Initialize the application
 */
function initializeApp() {
    // Initialize tooltips
    initializeTooltips();
    
    // Initialize form validation
    initializeFormValidation();
    
    // Initialize date formatting
    initializeDateFormatting();
    
    // Initialize animations
    initializeAnimations();
    
    // Initialize utility functions
    initializeUtilities();
    
    // Initialize modals
    initializeModals();
    
    // Log initialization
    console.log('IVF Journey Tracker initialized successfully');
}

/**
 * Initialize Bootstrap tooltips
 */
function initializeTooltips() {
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Initialize form validation
 */
function initializeFormValidation() {
    // Add Bootstrap validation classes
    const forms = document.querySelectorAll('.needs-validation');
    
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });
    
    // Custom validation for specific forms
    initializePasswordValidation();
    initializeBMICalculation();
    initializeDateValidation();
}

/**
 * Password confirmation validation
 */
function initializePasswordValidation() {
    const confirmPasswordInput = document.getElementById('confirm_password');
    const passwordInput = document.getElementById('password');
    
    if (confirmPasswordInput && passwordInput) {
        confirmPasswordInput.addEventListener('input', function() {
            if (passwordInput.value !== confirmPasswordInput.value) {
                confirmPasswordInput.setCustomValidity('Passwords do not match');
            } else {
                confirmPasswordInput.setCustomValidity('');
            }
        });
        
        passwordInput.addEventListener('input', function() {
            if (confirmPasswordInput.value) {
                confirmPasswordInput.dispatchEvent(new Event('input'));
            }
        });
    }
}

/**
 * BMI calculation for profile forms
 */
function initializeBMICalculation() {
    const heightInput = document.getElementById('height');
    const weightInput = document.getElementById('weight');
    const bmiInput = document.getElementById('bmi');
    const bmiCategory = document.getElementById('bmiCategory');
    
    if (heightInput && weightInput && bmiInput) {
        function calculateBMI() {
            const height = parseFloat(heightInput.value);
            const weight = parseFloat(weightInput.value);
            
            if (height && weight) {
                const heightInMeters = height / 100;
                const bmi = weight / (heightInMeters * heightInMeters);
                bmiInput.value = bmi.toFixed(1);
                
                // Update category
                if (bmiCategory) {
                    let category = '';
                    let categoryClass = '';
                    
                    if (bmi < 18.5) {
                        category = 'Underweight';
                        categoryClass = 'text-warning';
                    } else if (bmi < 25) {
                        category = 'Normal';
                        categoryClass = 'text-success';
                    } else if (bmi < 30) {
                        category = 'Overweight';
                        categoryClass = 'text-warning';
                    } else {
                        category = 'Obese';
                        categoryClass = 'text-danger';
                    }
                    
                    bmiCategory.textContent = category;
                    bmiCategory.className = `input-group-text ${categoryClass}`;
                }
            }
        }
        
        heightInput.addEventListener('input', calculateBMI);
        weightInput.addEventListener('input', calculateBMI);
        
        // Calculate on page load if values exist
        calculateBMI();
    }
}

/**
 * Date validation
 */
function initializeDateValidation() {
    const dateInputs = document.querySelectorAll('input[type="date"]');
    
    dateInputs.forEach(input => {
        input.addEventListener('change', function() {
            const selectedDate = new Date(this.value);
            const today = new Date();
            
            // Check if date is in the future (for birth dates)
            if (this.name === 'date_of_birth' && selectedDate > today) {
                this.setCustomValidity('Birth date cannot be in the future');
            } else {
                this.setCustomValidity('');
            }
        });
    });
}

/**
 * Initialize date formatting
 */
function initializeDateFormatting() {
    // Format all date elements with class 'format-date'
    const dateElements = document.querySelectorAll('.format-date');
    
    dateElements.forEach(element => {
        const dateValue = element.textContent || element.value;
        if (dateValue) {
            const date = new Date(dateValue);
            if (!isNaN(date.getTime())) {
                element.textContent = formatDate(date);
            }
        }
    });
    
    // Set current date where needed
    const currentDateElements = document.querySelectorAll('#currentDate');
    currentDateElements.forEach(element => {
        element.textContent = new Date().toLocaleDateString('en-US', {
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
    });
}

/**
 * Format date helper function
 */
function formatDate(date, options = {}) {
    const defaultOptions = {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    };
    
    return date.toLocaleDateString('en-US', { ...defaultOptions, ...options });
}

/**
 * Initialize animations
 */
function initializeAnimations() {
    // Intersection Observer for scroll animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-fade-in');
            }
        });
    }, observerOptions);
    
    // Observe elements with animation classes
    const animatedElements = document.querySelectorAll('.animate-on-scroll');
    animatedElements.forEach(el => observer.observe(el));
    
    // Counter animations for statistics
    initializeCounterAnimations();
}

/**
 * Initialize counter animations for statistics
 */
function initializeCounterAnimations() {
    const counters = document.querySelectorAll('.counter');
    
    counters.forEach(counter => {
        const target = parseInt(counter.getAttribute('data-target') || counter.textContent);
        const duration = 2000; // 2 seconds
        const increment = target / (duration / 16); // 60fps
        let current = 0;
        
        const updateCounter = () => {
            if (current < target) {
                current += increment;
                counter.textContent = Math.floor(current);
                requestAnimationFrame(updateCounter);
            } else {
                counter.textContent = target;
            }
        };
        
        // Start animation when element is in view
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    updateCounter();
                    observer.unobserve(counter);
                }
            });
        });
        
        observer.observe(counter);
    });
}

/**
 * Initialize utility functions
 */
function initializeUtilities() {
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Auto-hide alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            if (alert.parentNode) {
                alert.style.transition = 'opacity 0.5s ease-out';
                alert.style.opacity = '0';
                setTimeout(() => {
                    if (alert.parentNode) {
                        alert.remove();
                    }
                }, 500);
            }
        }, 5000);
    });
    
    // Initialize copy to clipboard functionality
    initializeClipboard();
    
    // Initialize file upload previews
    initializeFileUploads();
    
    // Initialize range input displays
    initializeRangeInputs();

}

/**
 * Initialize clipboard functionality
 */
function initializeClipboard() {
    const copyButtons = document.querySelectorAll('[data-copy]');
    
    copyButtons.forEach(button => {
        button.addEventListener('click', function() {
            const textToCopy = this.getAttribute('data-copy');
            
            navigator.clipboard.writeText(textToCopy).then(() => {
                showToast('Copied to clipboard!', 'success');
            }).catch(err => {
                console.error('Failed to copy text: ', err);
                showToast('Failed to copy text', 'error');
            });
        });
    });
}

/**
 * Initialize file upload previews
 */
function initializeFileUploads() {
    const fileInputs = document.querySelectorAll('input[type="file"]');
    
    fileInputs.forEach(input => {
        input.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                // Show file info
                const fileInfo = document.createElement('div');
                fileInfo.className = 'file-info mt-2';
                fileInfo.innerHTML = `
                    <small class="text-muted">
                        <i class="fas fa-file me-2"></i>
                        ${file.name} (${formatFileSize(file.size)})
                    </small>
                `;
                
                // Remove existing file info
                const existingInfo = this.parentNode.querySelector('.file-info');
                if (existingInfo) {
                    existingInfo.remove();
                }
                
                // Add new file info
                this.parentNode.appendChild(fileInfo);
            }
        });
    });
}

/**
 * Initialize range inputs
 */
function initializeRangeInputs() {
    const rangeInputs = document.querySelectorAll('input[type="range"]');
    
    rangeInputs.forEach(input => {
        const updateValue = () => {
            const valueDisplay = document.getElementById(input.id + 'Value');
            if (valueDisplay) {
                valueDisplay.textContent = input.value;
            }
            
            // Update progress bar if exists
            const progress = (input.value - input.min) / (input.max - input.min) * 100;
            input.style.background = `linear-gradient(to right, var(--primary-color) 0%, var(--primary-color) ${progress}%, #ddd ${progress}%, #ddd 100%)`;
        };
        
        input.addEventListener('input', updateValue);
        updateValue(); // Initialize
    });
}

/**
 * Initialize modals
 */
function initializeModals() {
    // Auto-focus on modal inputs when shown
    const modals = document.querySelectorAll('.modal');
    
    modals.forEach(modal => {
        modal.addEventListener('shown.bs.modal', function() {
            const firstInput = this.querySelector('input, textarea, select');
            if (firstInput) {
                firstInput.focus();
            }
        });
        
        // Clear form when modal is hidden
        modal.addEventListener('hidden.bs.modal', function() {
            const form = this.querySelector('form');
            if (form && form.getAttribute('data-clear-on-hide') !== 'false') {
                form.reset();
                form.classList.remove('was-validated');
            }
        });
    });
}

/**
 * Utility Functions
 */

/**
 * Show toast notification
 */
function showToast(message, type = 'info', duration = 3000) {
    const toast = document.createElement('div');
    toast.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    toast.style.cssText = `
        top: 20px;
        right: 20px;
        z-index: 9999;
        min-width: 300px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    `;
    
    toast.innerHTML = `
        <div class="d-flex align-items-center">
            <i class="fas fa-${getToastIcon(type)} me-2"></i>
            ${message}
            <button type="button" class="btn-close ms-auto" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    document.body.appendChild(toast);
    
    // Auto-remove after duration
    setTimeout(() => {
        if (toast.parentNode) {
            toast.remove();
        }
    }, duration);
    
    return toast;
}

/**
 * Get icon for toast type
 */
function getToastIcon(type) {
    const icons = {
        'success': 'check-circle',
        'error': 'exclamation-circle',
        'warning': 'exclamation-triangle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

/**
 * Format file size
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Debounce function for performance
 */
function debounce(func, wait, immediate) {
    let timeout;
    return function executedFunction() {
        const context = this;
        const args = arguments;
        
        const later = function() {
            timeout = null;
            if (!immediate) func.apply(context, args);
        };
        
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        
        if (callNow) func.apply(context, args);
    };
}

/**
 * Throttle function for performance
 */
function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    }
}

/**
 * Local storage helpers
 */
const Storage = {
    set: function(key, value) {
        try {
            localStorage.setItem(key, JSON.stringify(value));
        } catch (e) {
            console.error('Failed to save to localStorage:', e);
        }
    },
    
    get: function(key, defaultValue = null) {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : defaultValue;
        } catch (e) {
            console.error('Failed to get from localStorage:', e);
            return defaultValue;
        }
    },
    
    remove: function(key) {
        try {
            localStorage.removeItem(key);
        } catch (e) {
            console.error('Failed to remove from localStorage:', e);
        }
    }
};

/**
 * Form submission helpers
 */
function submitFormWithLoading(form, submitButton) {
    const originalText = submitButton.innerHTML;
    const loadingText = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
    
    submitButton.innerHTML = loadingText;
    submitButton.disabled = true;
    
    // Return a function to reset the button
    return function() {
        submitButton.innerHTML = originalText;
        submitButton.disabled = false;
    };
}

/**
 * API request helper
 */
async function apiRequest(url, options = {}) {
    const defaultOptions = {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        }
    };
    
    try {
        const response = await fetch(url, { ...defaultOptions, ...options });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
}

/**
 * Chart initialization helper
 */
function initializeChart(canvasId, config) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.error(`Canvas with id '${canvasId}' not found`);
        return null;
    }
    
    // Default chart options
    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'bottom',
                labels: {
                    padding: 20,
                    usePointStyle: true
                }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                grid: {
                    color: 'rgba(0, 0, 0, 0.1)'
                }
            },
            x: {
                grid: {
                    color: 'rgba(0, 0, 0, 0.1)'
                }
            }
        }
    };
    
    // Merge options
    if (config.options) {
        config.options = { ...defaultOptions, ...config.options };
    } else {
        config.options = defaultOptions;
    }
    
    return new Chart(canvas.getContext('2d'), config);
}

/**
 * Wellness data visualization helper
 */
function createWellnessChart(canvasId, data) {
    return initializeChart(canvasId, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: [{
                label: 'Mood',
                data: data.mood,
                borderColor: '#0d6efd',
                backgroundColor: 'rgba(13, 110, 253, 0.1)',
                tension: 0.3,
                fill: true
            }, {
                label: 'Stress',
                data: data.stress,
                borderColor: '#ffc107',
                backgroundColor: 'rgba(255, 193, 7, 0.1)',
                tension: 0.3,
                fill: true
            }, {
                label: 'Sleep Quality',
                data: data.sleep_quality,
                borderColor: '#198754',
                backgroundColor: 'rgba(25, 135, 84, 0.1)',
                tension: 0.3,
                fill: true
            }, {
                label: 'Energy',
                data: data.energy,
                borderColor: '#dc3545',
                backgroundColor: 'rgba(220, 53, 69, 0.1)',
                tension: 0.3,
                fill: true
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    max: 5,
                    ticks: {
                        stepSize: 1
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.raw}/5`;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Error handling
 */
window.addEventListener('error', function(e) {
    console.error('JavaScript error:', e.error);

    // Only show toast for critical errors, not all JS errors
    if (e.error && e.error.message && (
        e.error.message.includes('TypeError') ||
        e.error.message.includes('ReferenceError') ||
        e.error.message.includes('SyntaxError')
    )) {
        showToast('An error occurred. Please refresh the page.', 'error', 5000);
    }
});

// Unhandled promise rejections
window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled promise rejection:', e.reason);
    showToast('An error occurred while processing your request.', 'error', 5000);
});

/**
 * Export utilities for use in other scripts
 */
window.IVFTracker = {
    showToast,
    formatFileSize,
    formatDate,
    debounce,
    throttle,
    Storage,
    apiRequest,
    initializeChart,
    createWellnessChart,
    submitFormWithLoading
};
