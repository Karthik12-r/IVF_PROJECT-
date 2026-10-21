import csv
import random
from main import app
from database import db
from models import Clinic

# Approximate coordinates for major Indian cities (latitude, longitude)
city_coordinates = {
    "Mumbai": (19.0760, 72.8777), "Delhi": (28.7041, 77.1025), "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707), "Kolkata": (22.5726, 88.3639), "Pune": (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714), "Jaipur": (26.9124, 75.7873), "Hyderabad": (17.3850, 78.4867),
    "Chandigarh": (30.7333, 76.7794), "Lucknow": (26.8467, 80.9462), "Indore": (22.7196, 75.8577),
    "Bhopal": (23.2599, 77.4126), "Patna": (25.5941, 85.1376), "Nagpur": (21.1458, 79.0882),
    "Vadodara": (22.3072, 73.1812), "Ludhiana": (30.9010, 75.8573), "Agra": (27.1767, 78.0081),
    "Nashik": (19.9975, 73.7898), "Faridabad": (28.4089, 77.3178), "Meerut": (28.9845, 77.7064),
    "Rajkot": (22.3039, 70.8022), "Varanasi": (25.3176, 82.9739), "Srinagar": (34.0837, 74.7973),
    "Aurangabad": (19.8762, 75.3433), "Dhanbad": (23.7957, 86.4304), "Amritsar": (31.6340, 74.8723),
    "Navi Mumbai": (19.0330, 73.0297), "Allahabad": (25.4358, 81.8463), "Ranchi": (23.3441, 85.3096),
    "Howrah": (22.5958, 88.2636), "Coimbatore": (11.0168, 76.9558), "Jabalpur": (23.1815, 79.9864),
    "Gwalior": (26.2183, 78.1828), "Vijayawada": (16.5062, 80.6480), "Jodhpur": (26.2389, 73.0243),
    "Madurai": (9.9252, 78.1198), "Raipur": (21.2514, 81.6296), "Kota": (25.2138, 75.8648),
    "Guwahati": (26.1445, 91.7362), "Chandrapur": (19.9615, 79.2961), "Solapur": (17.6599, 75.9064),
    "Tiruchirappalli": (10.7905, 78.7047), "Bareilly": (28.3670, 79.4304), "Moradabad": (28.8386, 78.7733),
    "Mysore": (12.2958, 76.6394), "Tiruppur": (11.1085, 77.3411), "Gurgaon": (28.4595, 77.0266),
    "Aligarh": (27.8974, 78.0880), "Jalandhar": (31.3260, 75.5762), "Bhubaneswar": (20.2961, 85.8245),
    "Salem": (11.6643, 78.1460), "Warangal": (17.9784, 79.5941), "Guntur": (16.3067, 80.4365),
    "Bhiwandi": (19.2813, 73.0483), "Saharanpur": (29.9679, 77.5510), "Gorakhpur": (26.7606, 83.3732),
    "Bikaner": (28.0229, 73.3119), "Amravati": (20.9374, 77.7796), "Noida": (28.5355, 77.3910),
    "Jamshedpur": (22.8046, 86.2029), "Bhilai": (21.1938, 81.3509), "Cuttack": (20.4625, 85.8830),
    "Firozabad": (27.1592, 78.3958), "Kochi": (9.9312, 76.2673), "Nellore": (14.4426, 79.9865),
    "Bhavnagar": (21.7645, 72.1519), "Dehradun": (30.3165, 78.0322), "Durgapur": (23.5204, 87.3119),
    "Asansol": (23.6739, 86.9524), "Rourkela": (22.2604, 84.8536), "Nanded": (19.1383, 77.3210),
    "Kolhapur": (16.7050, 74.2433), "Ajmer": (26.4499, 74.6399), "Akola": (20.7002, 77.0082),
    "Gulbarga": (17.3297, 76.8343), "Jamnagar": (22.4707, 70.0577), "Ujjain": (23.1765, 75.7885),
    "Loni": (28.7528, 77.2880), "Siliguri": (26.7271, 88.3953), "Jhansi": (25.4484, 78.5685),
    "Ulhasnagar": (19.2215, 73.1645), "Jammu": (32.7266, 74.8570), "Sangli": (16.8544, 74.5642),
    "Erode": (11.3410, 77.7172), "Belgaum": (15.8497, 74.4977), "Mangalore": (12.9141, 74.8560),
    "Ambattur": (13.1143, 80.1548), "Tirunelveli": (8.7139, 77.7567), "Malegaon": (20.5537, 74.5288),
    "Gaya": (24.7914, 85.0002), "Thiruvananthapuram": (8.5241, 76.9366), "Udaipur": (24.5854, 73.7125),
    "Maheshtala": (22.5086, 88.2532), "Davanagere": (14.4644, 75.9218), "Kozhikode": (11.2588, 75.7804),
    "Kurnool": (15.8281, 78.0373), "Rajpur Sonarpur": (22.4490, 88.3915), "Rajahmundry": (17.0005, 81.8040),
    "Bokaro": (23.6693, 86.1511), "South Dumdum": (22.6100, 88.4000), "Bellary": (15.1394, 76.9214),
    "Patiala": (30.3398, 76.3869), "Gopalpur": (22.6142, 88.3911), "Agartala": (23.8315, 91.2868),
    "Bhagalpur": (25.2425, 87.0296), "Muzaffarnagar": (29.4727, 77.7085), "Bhatpara": (22.8714, 88.4089),
    "Panihati": (22.6940, 88.3744), "Latur": (18.4088, 76.5604), "Dhule": (20.9042, 74.7749),
    "Tirupati": (13.6288, 79.4192), "Rohtak": (28.8955, 76.6066), "Korba": (22.3595, 82.7501),
    "Bhilwara": (25.3214, 74.5885), "Berhampur": (19.3149, 84.7941), "Muzaffarpur": (26.1209, 85.3647),
    "Ahmednagar": (19.0952, 74.7496), "Mathura": (27.4924, 77.6737), "Kollam": (8.8932, 76.6141),
    "Avadi": (13.1067, 80.0970), "Kadapa": (14.4674, 78.8242), "Anantapur": (14.6819, 77.6006),
    "Kamarhati": (22.6711, 88.3747), "Bilaspur": (22.0797, 82.1391), "Sambalpur": (21.4669, 83.9812),
    "Shahjahanpur": (27.8815, 79.9123), "Satara": (17.6805, 74.0183), "Bijapur": (16.8302, 75.7100),
    "Rampur": (28.7896, 79.0249), "Shorapur": (16.5210, 76.7574), "Nagarcoil": (8.1773, 77.4344),
    "Chittoor": (13.2172, 79.1003), "Panipat": (29.3909, 76.9635), "Darbhanga": (26.1542, 85.8918),
    "Biharsharif": (25.1982, 85.5214), "Sikar": (27.6094, 75.1399), "Ozhukarai": (11.9489, 79.7924),
    "Mirzapur": (25.1337, 82.5644), "Karnal": (29.6857, 76.9905), "Ballia": (25.7584, 84.1487),
    "Sonipat": (28.9286, 77.0915), "Farrukhabad": (27.3826, 79.5941), "Sagar": (23.8388, 78.7378),
    "Durg": (21.1904, 81.2849), "Imphal": (24.8170, 93.9368), "Ratlam": (23.3342, 75.0370),
    "Hapur": (28.7306, 77.7759), "Arrah": (25.5560, 84.6603), "Karimnagar": (18.4386, 79.1288),
    "Anantnag": (33.7311, 75.1487), "Etawah": (26.7769, 79.0234), "Ambernath": (19.2088, 73.1860),
    "North Dumdum": (22.6620, 88.4194), "Bharatpur": (27.2173, 77.4895), "Begusarai": (25.4167, 86.1339),
    "New Delhi": (28.6139, 77.2090), "Gandhidham": (23.0753, 70.1337), "Baranagar": (22.6433, 88.3654),
    "Tiruvottiyur": (13.1646, 80.3041), "Pondicherry": (11.9139, 79.8145), "Thoothukudi": (8.7642, 78.1348),
    "Rewa": (24.5373, 81.3042), "Raichur": (16.2120, 77.3439), "Pali": (25.7711, 73.3231),
    "Ramagundam": (18.7550, 79.4740), "Haridwar": (29.9457, 78.1642), "Vijayanagaram": (18.1067, 83.3956),
    "Tenali": (16.2420, 80.6400), "Sri Ganganagar": (29.9094, 73.8801), "Karawal Nagar": (28.7283, 77.2767),
    "Mango": (22.8327, 86.2194), "Thanjavur": (10.7870, 79.1378), "Bulandshahr": (28.4060, 77.8498),
    "Uluberia": (22.4744, 88.1000), "Katni": (23.8343, 80.3948), "Sambhal": (28.5841, 78.5696),
    "Singrauli": (24.1992, 82.6645), "Nadiad": (22.6916, 72.8634), "Secunderabad": (17.4399, 78.4983),
    "Naihati": (22.8940, 88.4249), "Yamunanagar": (30.1290, 77.2674), "Bidhan Nagar": (22.5867, 88.4172),
    "Pallavaram": (12.9675, 80.1491), "Bidar": (17.9133, 77.5301), "Munger": (25.3748, 86.4735),
    "Panchkula": (30.6942, 76.8606), "Burhanpur": (21.3090, 76.2300), "Kharagpur": (22.3460, 87.2310),
    "Dindigul": (10.3673, 77.9803), "Gandhinagar": (23.2156, 72.6369), "Hospet": (15.2695, 76.3871),
    "Nangloi Jat": (28.6848, 77.0678), "Malda": (25.0108, 88.1411), "Ongole": (15.5057, 80.0499),
    "Deoghar": (24.4850, 86.6923), "Chapra": (25.7803, 84.7471), "Haldwani": (29.2183, 79.5127),
    "Khandwa": (21.8257, 76.3521), "Nandyal": (15.4786, 78.4836), "Morena": (26.4934, 77.9905),
    "Amroha": (28.9044, 78.4671), "Anand": (22.5645, 72.9289), "Bhind": (26.5637, 78.7871),
    "Bhalswa Jahangir Pur": (28.7354, 77.1638), "Madhyamgram": (22.6894, 88.4459), "Bhiwani": (28.7971, 76.1335),
    "Visakhapatnam": (17.6868, 83.2185), "Navsari": (20.9467, 72.9520), "Bahraich": (27.5708, 81.5980),
    "Vellore": (12.9165, 79.1325), "Mahesana": (23.5880, 72.3693), "Raebareli": (26.2345, 81.2409)
}

def get_coordinates(city):
    """Get approximate coordinates for a city, with small random variation"""
    if city in city_coordinates:
        lat, lng = city_coordinates[city]
        # Add small random variation (±0.01 degrees ≈ 1km)
        lat += random.uniform(-0.01, 0.01)
        lng += random.uniform(-0.01, 0.01)
        return lat, lng
    else:
        # Default to center of India with variation
        return 20.5937 + random.uniform(-5, 5), 78.9629 + random.uniform(-5, 5)

def import_clinics_from_csv(csv_file='indian_ivf_clinics.csv'):
    with app.app_context():
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            clinics_added = 0
            for row in reader:
                # Check if clinic already exists by name
                existing_clinic = Clinic.query.filter_by(name=row['Clinic Name']).first()
                if existing_clinic:
                    print(f"Clinic '{row['Clinic Name']}' already exists, skipping.")
                    continue

                # Get coordinates for the city
                lat, lng = get_coordinates(row['City'])

                # Create new clinic
                clinic = Clinic(
                    name=row['Clinic Name'],
                    address=row['Address'],
                    city=row['City'],
                    state=row['State'],
                    phone=row['Phone'],
                    website=row['Website'],
                    description=row['Description'],
                    latitude=lat,
                    longitude=lng
                )
                db.session.add(clinic)
                clinics_added += 1
                print(f"Added clinic: {row['Clinic Name']} (lat: {lat:.4f}, lng: {lng:.4f})")

            db.session.commit()
            print(f"Successfully imported {clinics_added} clinics from {csv_file}")

if __name__ == "__main__":
    import_clinics_from_csv()
