// Find Clinic JavaScript functionality

// Global variables for Google Maps
let googleMap;
let markers = [];
let selectedLocation = null;

// Initialize Google Places Autocomplete
function initAutocomplete() {
    const queryInput = document.getElementById('query-input');
    const locationDisplay = document.getElementById('location-display');
    const selectedLocationSpan = document.getElementById('selected-location');

    if (!queryInput) return;

    // Initialize Google Places Autocomplete
    const autocomplete = new google.maps.places.Autocomplete(queryInput, {
        types: ['geocode'], // Restrict to geographical locations
        componentRestrictions: { country: 'in' } // Restrict to India
    });

    // Listen for place selection
    autocomplete.addListener('place_changed', function() {
        const place = autocomplete.getPlace();

        if (!place.geometry) {
            console.log("No details available for input: '" + place.name + "'");
            return;
        }

        // Store selected location
        selectedLocation = {
            lat: place.geometry.location.lat(),
            lng: place.geometry.location.lng(),
            address: place.formatted_address,
            name: place.name
        };

        // Display selected location
        selectedLocationSpan.textContent = place.formatted_address;
        locationDisplay.style.display = 'block';

        // Update map if it exists
        if (googleMap) {
            googleMap.setCenter(selectedLocation);
            googleMap.setZoom(15);

            // Clear existing markers
            markers.forEach(marker => marker.setMap(null));
            markers = [];

            // Add marker for selected location
            const marker = new google.maps.Marker({
                position: selectedLocation,
                map: googleMap,
                title: selectedLocation.name
            });
            markers.push(marker);
        }

        console.log('Selected location:', selectedLocation);
    });

    // Clear location display when input is cleared
    queryInput.addEventListener('input', function() {
        if (this.value === '') {
            locationDisplay.style.display = 'none';
            selectedLocation = null;
        }
    });
}

// Initialize Google Maps
function initMap() {
    // Default center (India)
    const defaultCenter = { lat: 20.5937, lng: 78.9629 };
    const defaultZoom = 5;

    // Initialize Google Map
    googleMap = new google.maps.Map(document.getElementById('google-map'), {
        center: defaultCenter,
        zoom: defaultZoom,
        mapTypeControl: true,
        streetViewControl: true,
        fullscreenControl: true
    });

    // Get clinic data from the page (passed from template)
    const clinicsData = window.clinicsData || [];

    // Add markers for clinics with coordinates
    clinicsData.forEach(clinic => {
        if (clinic.latitude && clinic.longitude) {
            const marker = new google.maps.Marker({
                position: { lat: clinic.latitude, lng: clinic.longitude },
                map: googleMap,
                title: clinic.name
            });

            // Create info window content
            const infoWindowContent = `
                <div class="clinic-info-window">
                    <h6 class="mb-2">${clinic.name}</h6>
                    <p class="mb-1"><i class="fas fa-map-marker-alt me-1"></i>${clinic.city}, ${clinic.state}</p>
                    ${clinic.phone ? `<p class="mb-1"><i class="fas fa-phone me-1"></i>${clinic.phone}</p>` : ''}
                    ${clinic.website ? `<p class="mb-1"><i class="fas fa-globe me-1"></i><a href="${clinic.website}" target="_blank">Website</a></p>` : ''}
                    <a href="/clinic/${clinic.id}" class="btn btn-success btn-sm mt-2">View Details</a>
                </div>
            `;

            const infoWindow = new google.maps.InfoWindow({
                content: infoWindowContent
            });

            marker.addListener('click', () => {
                infoWindow.open(googleMap, marker);
                googleMap.setCenter(marker.getPosition());
                googleMap.setZoom(15);
            });

            markers.push(marker);
        }
    });

    // Fit map to show all markers if there are any
    if (markers.length > 0) {
        const bounds = new google.maps.LatLngBounds();
        markers.forEach(marker => bounds.extend(marker.getPosition()));
        googleMap.fitBounds(bounds);

        // Don't zoom in too much if there's only one marker
        google.maps.event.addListenerOnce(googleMap, 'bounds_changed', function() {
            if (googleMap.getZoom() > 15) {
                googleMap.setZoom(15);
            }
        });
    }

    return googleMap;
}

// Add some basic CSS for autocomplete
const style = document.createElement('style');
style.textContent = `
    .autocomplete-items {
        position: absolute;
        border: 1px solid #d4d4d4;
        border-bottom: none;
        border-top: none;
        z-index: 99;
        top: 100%;
        left: 0;
        right: 0;
        background-color: white;
        max-height: 200px;
        overflow-y: auto;
    }
    .autocomplete-items div {
        padding: 10px;
        cursor: pointer;
        background-color: #fff;
        border-bottom: 1px solid #d4d4d4;
    }
    .autocomplete-items div:hover {
        background-color: #e9e9e9;
    }
    .autocomplete-active {
        background-color: #1e90ff !important;
        color: #ffffff !important;
    }
`;
document.head.appendChild(style);
