import csv
import random

# Lists of realistic Indian cities, states, and clinic name components
cities_states = [
    ("Mumbai", "Maharashtra"), ("Delhi", "Delhi"), ("Bengaluru", "Karnataka"),
    ("Chennai", "Tamil Nadu"), ("Kolkata", "West Bengal"), ("Pune", "Maharashtra"),
    ("Ahmedabad", "Gujarat"), ("Jaipur", "Rajasthan"), ("Hyderabad", "Telangana"),
    ("Chandigarh", "Chandigarh"), ("Lucknow", "Uttar Pradesh"), ("Indore", "Madhya Pradesh"),
    ("Bhopal", "Madhya Pradesh"), ("Patna", "Bihar"), ("Nagpur", "Maharashtra"),
    ("Vadodara", "Gujarat"), ("Ludhiana", "Punjab"), ("Agra", "Uttar Pradesh"),
    ("Nashik", "Maharashtra"), ("Faridabad", "Haryana"), ("Meerut", "Uttar Pradesh"),
    ("Rajkot", "Gujarat"), ("Varanasi", "Uttar Pradesh"), ("Srinagar", "Jammu and Kashmir"),
    ("Aurangabad", "Maharashtra"), ("Dhanbad", "Jharkhand"), ("Amritsar", "Punjab"),
    ("Navi Mumbai", "Maharashtra"), ("Allahabad", "Uttar Pradesh"), ("Ranchi", "Jharkhand"),
    ("Howrah", "West Bengal"), ("Coimbatore", "Tamil Nadu"), ("Jabalpur", "Madhya Pradesh"),
    ("Gwalior", "Madhya Pradesh"), ("Vijayawada", "Andhra Pradesh"), ("Jodhpur", "Rajasthan"),
    ("Madurai", "Tamil Nadu"), ("Raipur", "Chhattisgarh"), ("Kota", "Rajasthan"),
    ("Guwahati", "Assam"), ("Chandrapur", "Maharashtra"), ("Solapur", "Maharashtra"),
    ("Tiruchirappalli", "Tamil Nadu"), ("Bareilly", "Uttar Pradesh"), ("Moradabad", "Uttar Pradesh"),
    ("Mysore", "Karnataka"), ("Tiruppur", "Tamil Nadu"), ("Gurgaon", "Haryana"),
    ("Aligarh", "Uttar Pradesh"), ("Jalandhar", "Punjab"), ("Bhubaneswar", "Odisha"),
    ("Salem", "Tamil Nadu"), ("Warangal", "Telangana"), ("Guntur", "Andhra Pradesh"),
    ("Bhiwandi", "Maharashtra"), ("Saharanpur", "Uttar Pradesh"), ("Gorakhpur", "Uttar Pradesh"),
    ("Bikaner", "Rajasthan"), ("Amravati", "Maharashtra"), ("Noida", "Uttar Pradesh"),
    ("Jamshedpur", "Jharkhand"), ("Bhilai", "Chhattisgarh"), ("Cuttack", "Odisha"),
    ("Firozabad", "Uttar Pradesh"), ("Kochi", "Kerala"), ("Nellore", "Andhra Pradesh"),
    ("Bhavnagar", "Gujarat"), ("Dehradun", "Uttarakhand"), ("Durgapur", "West Bengal"),
    ("Asansol", "West Bengal"), ("Rourkela", "Odisha"), ("Nanded", "Maharashtra"),
    ("Kolhapur", "Maharashtra"), ("Ajmer", "Rajasthan"), ("Akola", "Maharashtra"),
    ("Gulbarga", "Karnataka"), ("Jamnagar", "Gujarat"), ("Ujjain", "Madhya Pradesh"),
    ("Loni", "Uttar Pradesh"), ("Siliguri", "West Bengal"), ("Jhansi", "Uttar Pradesh"),
    ("Ulhasnagar", "Maharashtra"), ("Jammu", "Jammu and Kashmir"), ("Sangli", "Maharashtra"),
    ("Erode", "Tamil Nadu"), ("Belgaum", "Karnataka"), ("Mangalore", "Karnataka"),
    ("Ambattur", "Tamil Nadu"), ("Tirunelveli", "Tamil Nadu"), ("Malegaon", "Maharashtra"),
    ("Gaya", "Bihar"), ("Thiruvananthapuram", "Kerala"), ("Udaipur", "Rajasthan"),
    ("Maheshtala", "West Bengal"), ("Davanagere", "Karnataka"), ("Kozhikode", "Kerala"),
    ("Kurnool", "Andhra Pradesh"), ("Rajpur Sonarpur", "West Bengal"), ("Rajahmundry", "Andhra Pradesh"),
    ("Bokaro", "Jharkhand"), ("South Dumdum", "West Bengal"), ("Bellary", "Karnataka"),
    ("Patiala", "Punjab"), ("Gopalpur", "West Bengal"), ("Agartala", "Tripura"),
    ("Bhagalpur", "Bihar"), ("Muzaffarnagar", "Uttar Pradesh"), ("Bhatpara", "West Bengal"),
    ("Panihati", "West Bengal"), ("Latur", "Maharashtra"), ("Dhule", "Maharashtra"),
    ("Tirupati", "Andhra Pradesh"), ("Rohtak", "Haryana"), ("Korba", "Chhattisgarh"),
    ("Bhilwara", "Rajasthan"), ("Berhampur", "Odisha"), ("Muzaffarpur", "Bihar"),
    ("Ahmednagar", "Maharashtra"), ("Mathura", "Uttar Pradesh"), ("Kollam", "Kerala"),
    ("Avadi", "Tamil Nadu"), ("Kadapa", "Andhra Pradesh"), ("Anantapur", "Andhra Pradesh"),
    ("Kamarhati", "West Bengal"), ("Bilaspur", "Chhattisgarh"), ("Sambalpur", "Odisha"),
    ("Shahjahanpur", "Uttar Pradesh"), ("Satara", "Maharashtra"), ("Bijapur", "Karnataka"),
    ("Rampur", "Uttar Pradesh"), ("Shorapur", "Karnataka"), ("Nagarcoil", "Tamil Nadu"),
    ("Chittoor", "Andhra Pradesh"), ("Panipat", "Haryana"), ("Darbhanga", "Bihar"),
    ("Biharsharif", "Bihar"), ("Sikar", "Rajasthan"), ("Ozhukarai", "Puducherry"),
    ("Mirzapur", "Uttar Pradesh"), ("Karnal", "Haryana"), ("Ballia", "Uttar Pradesh"),
    ("Patiala", "Punjab"), ("Sonipat", "Haryana"), ("Farrukhabad", "Uttar Pradesh"),
    ("Sagar", "Madhya Pradesh"), ("Rourkela", "Odisha"), ("Durg", "Chhattisgarh"),
    ("Imphal", "Manipur"), ("Ratlam", "Madhya Pradesh"), ("Hapur", "Uttar Pradesh"),
    ("Arrah", "Bihar"), ("Karimnagar", "Telangana"), ("Anantnag", "Jammu and Kashmir"),
    ("Etawah", "Uttar Pradesh"), ("Ambernath", "Maharashtra"), ("North Dumdum", "West Bengal"),
    ("Bharatpur", "Rajasthan"), ("Begusarai", "Bihar"), ("New Delhi", "Delhi"),
    ("Gandhidham", "Gujarat"), ("Baranagar", "West Bengal"), ("Tiruvottiyur", "Tamil Nadu"),
    ("Pondicherry", "Puducherry"), ("Sikar", "Rajasthan"), ("Thoothukudi", "Tamil Nadu"),
    ("Rewa", "Madhya Pradesh"), ("Mirzapur", "Uttar Pradesh"), ("Raichur", "Karnataka"),
    ("Pali", "Rajasthan"), ("Ramagundam", "Telangana"), ("Haridwar", "Uttarakhand"),
    ("Vijayanagaram", "Andhra Pradesh"), ("Tenali", "Andhra Pradesh"), ("Nagercoil", "Tamil Nadu"),
    ("Sri Ganganagar", "Rajasthan"), ("Karawal Nagar", "Delhi"), ("Mango", "Jharkhand"),
    ("Thanjavur", "Tamil Nadu"), ("Bulandshahr", "Uttar Pradesh"), ("Uluberia", "West Bengal"),
    ("Katni", "Madhya Pradesh"), ("Sambhal", "Uttar Pradesh"), ("Singrauli", "Madhya Pradesh"),
    ("Nadiad", "Gujarat"), ("Secunderabad", "Telangana"), ("Naihati", "West Bengal"),
    ("Yamunanagar", "Haryana"), ("Bidhan Nagar", "West Bengal"), ("Pallavaram", "Tamil Nadu"),
    ("Bidar", "Karnataka"), ("Munger", "Bihar"), ("Panchkula", "Haryana"),
    ("Burhanpur", "Madhya Pradesh"), ("Raurkela", "Odisha"), ("Kharagpur", "West Bengal"),
    ("Dindigul", "Tamil Nadu"), ("Gandhinagar", "Gujarat"), ("Hospet", "Karnataka"),
    ("Nangloi Jat", "Delhi"), ("Malda", "West Bengal"), ("Ongole", "Andhra Pradesh"),
    ("Deoghar", "Jharkhand"), ("Chapra", "Bihar"), ("Haldwani", "Uttarakhand"),
    ("Khandwa", "Madhya Pradesh"), ("Nandyal", "Andhra Pradesh"), ("Morena", "Madhya Pradesh"),
    ("Amroha", "Uttar Pradesh"), ("Anand", "Gujarat"), ("Bhind", "Madhya Pradesh"),
    ("Bhalswa Jahangir Pur", "Delhi"), ("Madhyamgram", "West Bengal"), ("Bhiwani", "Haryana"),
    ("Visakhapatnam", "Andhra Pradesh"), ("Navsari", "Gujarat"), ("Bahraich", "Uttar Pradesh"),
    ("Vellore", "Tamil Nadu"), ("Mahesana", "Gujarat"), ("Raebareli", "Uttar Pradesh"),
    ("Chennai", "Tamil Nadu"), ("Hyderabad", "Telangana"), ("Pune", "Maharashtra"),
    ("Surat", "Gujarat"), ("Kanpur", "Uttar Pradesh"), ("Nagpur", "Maharashtra"),
    ("Patna", "Bihar"), ("Vadodara", "Gujarat"), ("Ghaziabad", "Uttar Pradesh"),
    ("Ludhiana", "Punjab"), ("Agra", "Uttar Pradesh"), ("Nashik", "Maharashtra"),
    ("Faridabad", "Haryana"), ("Meerut", "Uttar Pradesh"), ("Rajkot", "Gujarat"),
    ("Kalyan-Dombivli", "Maharashtra"), ("Vasai-Virar", "Maharashtra"), ("Varanasi", "Uttar Pradesh"),
    ("Srinagar", "Jammu and Kashmir"), ("Aurangabad", "Maharashtra"), ("Dhanbad", "Jharkhand"),
    ("Amritsar", "Punjab"), ("Navi Mumbai", "Maharashtra"), ("Allahabad", "Uttar Pradesh"),
    ("Ranchi", "Jharkhand"), ("Howrah", "West Bengal"), ("Coimbatore", "Tamil Nadu"),
    ("Jabalpur", "Madhya Pradesh"), ("Gwalior", "Madhya Pradesh"), ("Vijayawada", "Andhra Pradesh"),
    ("Jodhpur", "Rajasthan"), ("Madurai", "Tamil Nadu"), ("Raipur", "Chhattisgarh"),
    ("Kota", "Rajasthan"), ("Guwahati", "Assam"), ("Chandrapur", "Maharashtra"),
    ("Solapur", "Maharashtra"), ("Tiruchirappalli", "Tamil Nadu"), ("Bareilly", "Uttar Pradesh"),
    ("Moradabad", "Uttar Pradesh"), ("Mysore", "Karnataka"), ("Tiruppur", "Tamil Nadu"),
    ("Gurgaon", "Haryana"), ("Aligarh", "Uttar Pradesh"), ("Jalandhar", "Punjab"),
    ("Bhubaneswar", "Odisha"), ("Salem", "Tamil Nadu"), ("Warangal", "Telangana"),
    ("Guntur", "Andhra Pradesh"), ("Bhiwandi", "Maharashtra"), ("Saharanpur", "Uttar Pradesh"),
    ("Gorakhpur", "Uttar Pradesh"), ("Bikaner", "Rajasthan"), ("Amravati", "Maharashtra"),
    ("Noida", "Uttar Pradesh"), ("Jamshedpur", "Jharkhand"), ("Bhilai", "Chhattisgarh"),
    ("Cuttack", "Odisha"), ("Firozabad", "Uttar Pradesh"), ("Kochi", "Kerala"),
    ("Nellore", "Andhra Pradesh"), ("Bhavnagar", "Gujarat"), ("Dehradun", "Uttarakhand"),
    ("Durgapur", "West Bengal"), ("Asansol", "West Bengal"), ("Rourkela", "Odisha"),
    ("Nanded", "Maharashtra"), ("Kolhapur", "Maharashtra"), ("Ajmer", "Rajasthan"),
    ("Akola", "Maharashtra"), ("Gulbarga", "Karnataka"), ("Jamnagar", "Gujarat"),
    ("Ujjain", "Madhya Pradesh"), ("Loni", "Uttar Pradesh"), ("Siliguri", "West Bengal"),
    ("Jhansi", "Uttar Pradesh"), ("Ulhasnagar", "Maharashtra"), ("Jammu", "Jammu and Kashmir"),
    ("Sangli", "Maharashtra"), ("Erode", "Tamil Nadu"), ("Belgaum", "Karnataka"),
    ("Mangalore", "Karnataka"), ("Ambattur", "Tamil Nadu"), ("Tirunelveli", "Tamil Nadu"),
    ("Malegaon", "Maharashtra"), ("Gaya", "Bihar"), ("Thiruvananthapuram", "Kerala"),
    ("Udaipur", "Rajasthan"), ("Maheshtala", "West Bengal"), ("Davanagere", "Karnataka"),
    ("Kozhikode", "Kerala"), ("Kurnool", "Andhra Pradesh"), ("Rajpur Sonarpur", "West Bengal"),
    ("Rajahmundry", "Andhra Pradesh"), ("Bokaro", "Jharkhand"), ("South Dumdum", "West Bengal"),
    ("Bellary", "Karnataka"), ("Patiala", "Punjab"), ("Gopalpur", "West Bengal"),
    ("Agartala", "Tripura"), ("Bhagalpur", "Bihar"), ("Muzaffarnagar", "Uttar Pradesh"),
    ("Bhatpara", "West Bengal"), ("Panihati", "West Bengal"), ("Latur", "Maharashtra"),
    ("Dhule", "Maharashtra"), ("Tirupati", "Andhra Pradesh"), ("Rohtak", "Haryana"),
    ("Korba", "Chhattisgarh"), ("Bhilwara", "Rajasthan"), ("Berhampur", "Odisha"),
    ("Muzaffarpur", "Bihar"), ("Ahmednagar", "Maharashtra"), ("Mathura", "Uttar Pradesh"),
    ("Kollam", "Kerala"), ("Avadi", "Tamil Nadu"), ("Kadapa", "Andhra Pradesh"),
    ("Anantapur", "Andhra Pradesh"), ("Kamarhati", "West Bengal"), ("Bilaspur", "Chhattisgarh"),
    ("Sambalpur", "Odisha"), ("Shahjahanpur", "Uttar Pradesh"), ("Satara", "Maharashtra"),
    ("Bijapur", "Karnataka"), ("Rampur", "Uttar Pradesh"), ("Shorapur", "Karnataka"),
    ("Nagarcoil", "Tamil Nadu"), ("Chittoor", "Andhra Pradesh"), ("Panipat", "Haryana"),
    ("Darbhanga", "Bihar"), ("Biharsharif", "Bihar"), ("Sikar", "Rajasthan"),
    ("Ozhukarai", "Puducherry"), ("Mirzapur", "Uttar Pradesh"), ("Karnal", "Haryana"),
    ("Ballia", "Uttar Pradesh"), ("Patiala", "Punjab"), ("Sonipat", "Haryana"),
    ("Farrukhabad", "Uttar Pradesh"), ("Sagar", "Madhya Pradesh"), ("Rourkela", "Odisha"),
    ("Durg", "Chhattisgarh"), ("Imphal", "Manipur"), ("Ratlam", "Madhya Pradesh"),
    ("Hapur", "Uttar Pradesh"), ("Arrah", "Bihar"), ("Karimnagar", "Telangana"),
    ("Anantnag", "Jammu and Kashmir"), ("Etawah", "Uttar Pradesh"), ("Ambernath", "Maharashtra"),
    ("North Dumdum", "West Bengal"), ("Bharatpur", "Rajasthan"), ("Begusarai", "Bihar"),
    ("New Delhi", "Delhi"), ("Gandhidham", "Gujarat"), ("Baranagar", "West Bengal"),
    ("Tiruvottiyur", "Tamil Nadu"), ("Pondicherry", "Puducherry"), ("Sikar", "Rajasthan"),
    ("Thoothukudi", "Tamil Nadu"), ("Rewa", "Madhya Pradesh"), ("Mirzapur", "Uttar Pradesh"),
    ("Raichur", "Karnataka"), ("Pali", "Rajasthan"), ("Ramagundam", "Telangana"),
    ("Haridwar", "Uttarakhand"), ("Vijayanagaram", "Andhra Pradesh"), ("Tenali", "Andhra Pradesh"),
    ("Nagercoil", "Tamil Nadu"), ("Sri Ganganagar", "Rajasthan"), ("Karawal Nagar", "Delhi"),
    ("Mango", "Jharkhand"), ("Thanjavur", "Tamil Nadu"), ("Bulandshahr", "Uttar Pradesh"),
    ("Uluberia", "West Bengal"), ("Katni", "Madhya Pradesh"), ("Sambhal", "Uttar Pradesh"),
    ("Singrauli", "Madhya Pradesh"), ("Nadiad", "Gujarat"), ("Secunderabad", "Telangana"),
    ("Naihati", "West Bengal"), ("Yamunanagar", "Haryana"), ("Bidhan Nagar", "West Bengal"),
    ("Pallavaram", "Tamil Nadu"), ("Bidar", "Karnataka"), ("Munger", "Bihar"),
    ("Panchkula", "Haryana"), ("Burhanpur", "Madhya Pradesh"), ("Raurkela", "Odisha"),
    ("Kharagpur", "West Bengal"), ("Dindigul", "Tamil Nadu"), ("Gandhinagar", "Gujarat"),
    ("Hospet", "Karnataka"), ("Nangloi Jat", "Delhi"), ("Malda", "West Bengal"),
    ("Ongole", "Andhra Pradesh"), ("Deoghar", "Jharkhand"), ("Chapra", "Bihar"),
    ("Haldwani", "Uttarakhand"), ("Khandwa", "Madhya Pradesh"), ("Nandyal", "Andhra Pradesh"),
    ("Morena", "Madhya Pradesh"), ("Amroha", "Uttar Pradesh"), ("Anand", "Gujarat"),
    ("Bhind", "Madhya Pradesh"), ("Bhalswa Jahangir Pur", "Delhi"), ("Madhyamgram", "West Bengal"),
    ("Bhiwani", "Haryana"), ("Visakhapatnam", "Andhra Pradesh"), ("Navsari", "Gujarat"),
    ("Bahraich", "Uttar Pradesh"), ("Vellore", "Tamil Nadu"), ("Mahesana", "Gujarat"),
    ("Raebareli", "Uttar Pradesh"), ("Chennai", "Tamil Nadu"), ("Hyderabad", "Telangana"),
    ("Pune", "Maharashtra"), ("Surat", "Gujarat"), ("Kanpur", "Uttar Pradesh"),
    ("Nagpur", "Maharashtra"), ("Patna", "Bihar"), ("Vadodara", "Gujarat"),
    ("Ghaziabad", "Uttar Pradesh"), ("Ludhiana", "Punjab"), ("Agra", "Uttar Pradesh"),
    ("Nashik", "Maharashtra"), ("Faridabad", "Haryana"), ("Meerut", "Uttar Pradesh"),
    ("Rajkot", "Gujarat"), ("Kalyan-Dombivli", "Maharashtra"), ("Vasai-Virar", "Maharashtra"),
    ("Varanasi", "Uttar Pradesh"), ("Srinagar", "Jammu and Kashmir"), ("Aurangabad", "Maharashtra"),
    ("Dhanbad", "Jharkhand"), ("Amritsar", "Punjab"), ("Navi Mumbai", "Maharashtra"),
    ("Allahabad", "Uttar Pradesh"), ("Ranchi", "Jharkhand"), ("Howrah", "West Bengal"),
    ("Coimbatore", "Tamil Nadu"), ("Jabalpur", "Madhya Pradesh"), ("Gwalior", "Madhya Pradesh"),
    ("Vijayawada", "Andhra Pradesh"), ("Jodhpur", "Rajasthan"), ("Madurai", "Tamil Nadu"),
    ("Raipur", "Chhattisgarh"), ("Kota", "Rajasthan"), ("Guwahati", "Assam"),
    ("Chandrapur", "Maharashtra"), ("Solapur", "Maharashtra"), ("Tiruchirappalli", "Tamil Nadu"),
    ("Bareilly", "Uttar Pradesh"), ("Moradabad", "Uttar Pradesh"), ("Mysore", "Karnataka"),
    ("Tiruppur", "Tamil Nadu"), ("Gurgaon", "Haryana"), ("Aligarh", "Uttar Pradesh"),
    ("Jalandhar", "Punjab"), ("Bhubaneswar", "Odisha"), ("Salem", "Tamil Nadu"),
    ("Warangal", "Telangana"), ("Guntur", "Andhra Pradesh"), ("Bhiwandi", "Maharashtra"),
    ("Saharanpur", "Uttar Pradesh"), ("Gorakhpur", "Uttar Pradesh"), ("Bikaner", "Rajasthan"),
    ("Amravati", "Maharashtra"), ("Noida", "Uttar Pradesh"), ("Jamshedpur", "Jharkhand"),
    ("Bhilai", "Chhattisgarh"), ("Cuttack", "Odisha"), ("Firozabad", "Uttar Pradesh"),
    ("Kochi", "Kerala"), ("Nellore", "Andhra Pradesh"), ("Bhavnagar", "Gujarat"),
    ("Dehradun", "Uttarakhand"), ("Durgapur", "West Bengal"), ("Asansol", "West Bengal"),
    ("Rourkela", "Odisha"), ("Nanded", "Maharashtra"), ("Kolhapur", "Maharashtra"),
    ("Ajmer", "Rajasthan"), ("Akola", "Maharashtra"), ("Gulbarga", "Karnataka"),
    ("Jamnagar", "Gujarat"), ("Ujjain", "Madhya Pradesh"), ("Loni", "Uttar Pradesh"),
    ("Siliguri", "West Bengal"), ("Jhansi", "Uttar Pradesh"), ("Ulhasnagar", "Maharashtra"),
    ("Jammu", "Jammu and Kashmir"), ("Sangli", "Maharashtra"), ("Erode", "Tamil Nadu"),
    ("Belgaum", "Karnataka"), ("Mangalore", "Karnataka"), ("Ambattur", "Tamil Nadu"),
    ("Tirunelveli", "Tamil Nadu"), ("Malegaon", "Maharashtra"), ("Gaya", "Bihar"),
    ("Thiruvananthapuram", "Kerala"), ("Udaipur", "Rajasthan"), ("Maheshtala", "West Bengal"),
    ("Davanagere", "Karnataka"), ("Kozhikode", "Kerala"), ("Kurnool", "Andhra Pradesh"),
    ("Rajpur Sonarpur", "West Bengal"), ("Rajahmundry", "Andhra Pradesh"), ("Bokaro", "Jharkhand"),
    ("South Dumdum", "West Bengal"), ("Bellary", "Karnataka"), ("Patiala", "Punjab"),
    ("Gopalpur", "West Bengal"), ("Agartala", "Tripura"), ("Bhagalpur", "Bihar"),
    ("Muzaffarnagar", "Uttar Pradesh"), ("Bhatpara", "West Bengal"), ("Panihati", "West Bengal"),
    ("Latur", "Maharashtra"), ("Dhule", "Maharashtra"), ("Tirupati", "Andhra Pradesh"),
    ("Rohtak", "Haryana"), ("Korba", "Chhattisgarh"), ("Bhilwara", "Rajasthan"),
    ("Berhampur", "Odisha"), ("Muzaffarpur", "Bihar"), ("Ahmednagar", "Maharashtra"),
    ("Mathura", "Uttar Pradesh"), ("Kollam", "Kerala"), ("Avadi", "Tamil Nadu"),
    ("Kadapa", "Andhra Pradesh"), ("Anantapur", "Andhra Pradesh"), ("Kamarhati", "West Bengal"),
    ("Bilaspur", "Chhattisgarh"), ("Sambalpur", "Odisha"), ("Shahjahanpur", "Uttar Pradesh"),
    ("Satara", "Maharashtra"), ("Bijapur", "Karnataka"), ("Rampur", "Uttar Pradesh"),
    ("Shorapur", "Karnataka"), ("Nagarcoil", "Tamil Nadu"), ("Chittoor", "Andhra Pradesh"),
    ("Panipat", "Haryana"), ("Darbhanga", "Bihar"), ("Biharsharif", "Bihar"),
    ("Sikar", "Rajasthan"), ("Ozhukarai", "Puducherry"), ("Mirzapur", "Uttar Pradesh"),
    ("Karnal", "Haryana"), ("Ballia", "Uttar Pradesh"), ("Patiala", "Punjab"),
    ("Sonipat", "Haryana"), ("Farrukhabad", "Uttar Pradesh"), ("Sagar", "Madhya Pradesh"),
    ("Rourkela", "Odisha"), ("Durg", "Chhattisgarh"), ("Imphal", "Manipur"),
    ("Ratlam", "Madhya Pradesh"), ("Hapur", "Uttar Pradesh"), ("Arrah", "Bihar"),
    ("Karimnagar", "Telangana"), ("Anantnag", "Jammu and Kashmir"), ("Etawah", "Uttar Pradesh"),
    ("Ambernath", "Maharashtra"), ("North Dumdum", "West Bengal"), ("Bharatpur", "Rajasthan"),
    ("Begusarai", "Bihar"), ("New Delhi", "Delhi"), ("Gandhidham", "Gujarat"),
    ("Baranagar", "West Bengal"), ("Tiruvottiyur", "Tamil Nadu"), ("Pondicherry", "Puducherry"),
    ("Sikar", "Rajasthan"), ("Thoothukudi", "Tamil Nadu"), ("Rewa", "Madhya Pradesh"),
    ("Mirzapur", "Uttar Pradesh"), ("Raichur", "Karnataka"), ("Pali", "Rajasthan"),
    ("Ramagundam", "Telangana"), ("Haridwar", "Uttarakhand"), ("Vijayanagaram", "Andhra Pradesh"),
    ("Tenali", "Andhra Pradesh"), ("Nagercoil", "Tamil Nadu"), ("Sri Ganganagar", "Rajasthan"),
    ("Karawal Nagar", "Delhi"), ("Mango", "Jharkhand"), ("Thanjavur", "Tamil Nadu"),
    ("Bulandshahr", "Uttar Pradesh"), ("Uluberia", "West Bengal"), ("Katni", "Madhya Pradesh"),
    ("Sambhal", "Uttar Pradesh"), ("Singrauli", "Madhya Pradesh"), ("Nadiad", "Gujarat"),
    ("Secunderabad", "Telangana"), ("Naihati", "West Bengal"), ("Yamunanagar", "Haryana"),
    ("Bidhan Nagar", "West Bengal"), ("Pallavaram", "Tamil Nadu"), ("Bidar", "Karnataka"),
    ("Munger", "Bihar"), ("Panchkula", "Haryana"), ("Burhanpur", "Madhya Pradesh"),
    ("Raurkela", "Odisha"), ("Kharagpur", "West Bengal"), ("Dindigul", "Tamil Nadu"),
    ("Gandhinagar", "Gujarat"), ("Hospet", "Karnataka"), ("Nangloi Jat", "Delhi"),
    ("Malda", "West Bengal"), ("Ongole", "Andhra Pradesh"), ("Deoghar", "Jharkhand"),
    ("Chapra", "Bihar"), ("Haldwani", "Uttarakhand"), ("Khandwa", "Madhya Pradesh"),
    ("Nandyal", "Andhra Pradesh"), ("Morena", "Madhya Pradesh"), ("Amroha", "Uttar Pradesh"),
    ("Anand", "Gujarat"), ("Bhind", "Madhya Pradesh"), ("Bhalswa Jahangir Pur", "Delhi"),
    ("Madhyamgram", "West Bengal"), ("Bhiwani", "Haryana"), ("Visakhapatnam", "Andhra Pradesh"),
    ("Navsari", "Gujarat"), ("Bahraich", "Uttar Pradesh"), ("Vellore", "Tamil Nadu"),
    ("Mahesana", "Gujarat"), ("Raebareli", "Uttar Pradesh")
]

clinic_prefixes = ["Nova", "Milann", "Oasis", "Vins", "Vir k", "Artemis", "Infertility Solutions", "Dr. Rama", "Bloom", "Akanksha", "Vani", "Shivani", "Ankur", "Life Line", "Morpheus", "Fertility First", "Hope IVF", "Genesis", "Miracle", "Care IVF", "Advanced Fertility", "Prime IVF", "Elite Fertility", "Excel IVF", "Superior IVF", "Premium Fertility", "Royal IVF", "Golden IVF", "Diamond Fertility", "Platinum IVF", "Crystal IVF", "Pearl Fertility", "Ruby IVF", "Sapphire Fertility", "Emerald IVF", "Top IVF", "Best Fertility", "Leading IVF", "Premier Fertility", "Ultimate IVF", "Supreme Fertility", "Excellent IVF", "Superb Fertility", "Outstanding IVF", "Remarkable Fertility", "Exceptional IVF", "Wonderful Fertility", "Amazing IVF", "Fantastic Fertility", "Great IVF", "Fine Fertility", "Good IVF", "Nice Fertility", "Perfect IVF", "Ideal Fertility", "Optimal IVF", "Prime Fertility", "Choice IVF", "Select Fertility", "Preferred IVF", "Chosen Fertility", "Elite IVF", "Superior Fertility", "Premium IVF", "Royal Fertility", "Golden IVF", "Diamond Fertility", "Platinum IVF", "Crystal IVF", "Pearl Fertility", "Ruby IVF", "Sapphire Fertility", "Emerald IVF", "Top IVF", "Best Fertility", "Leading IVF", "Premier Fertility", "Ultimate IVF", "Supreme Fertility", "Excellent IVF", "Superb Fertility", "Outstanding IVF", "Remarkable Fertility", "Exceptional IVF", "Wonderful Fertility", "Amazing IVF", "Fantastic Fertility", "Great IVF", "Fine Fertility", "Good IVF", "Nice Fertility", "Perfect IVF", "Ideal Fertility", "Optimal IVF", "Prime Fertility", "Choice IVF", "Select Fertility", "Preferred IVF", "Chosen Fertility", "Elite IVF", "Superior Fertility", "Premium IVF", "Royal Fertility", "Golden IVF", "Diamond Fertility", "Platinum IVF", "Crystal IVF", "Pearl Fertility", "Ruby IVF", "Sapphire Fertility", "Emerald IVF", "Top IVF", "Best Fertility", "Leading IVF", "Premier Fertility", "Ultimate IVF", "Supreme Fertility", "Excellent IVF", "Superb Fertility", "Outstanding IVF", "Remarkable Fertility", "Exceptional IVF", "Wonderful Fertility", "Amazing IVF", "Fantastic Fertility", "Great IVF", "Fine Fertility", "Good IVF", "Nice Fertility", "Perfect IVF", "Ideal Fertility", "Optimal IVF", "Prime Fertility", "Choice IVF", "Select Fertility", "Preferred IVF", "Chosen Fertility", "Elite IVF", "Superior Fertility", "Premium IVF", "Royal Fertility", "Golden IVF", "Diamond Fertility", "Platinum IVF", "Crystal IVF", "Pearl Fertility", "Ruby IVF", "Sapphire Fertility", "Emerald IVF", "Top IVF", "Best Fertility", "Leading IVF", "Premier Fertility", "Ultimate IVF", "Supreme Fertility", "Excellent IVF", "Superb Fertility", "Outstanding IVF", "Remarkable Fertility", "Exceptional IVF", "Wonderful Fertility", "Amazing IVF", "Fantastic Fertility", "Great IVF", "Fine Fertility", "Good IVF", "Nice Fertility", "Perfect IVF", "Ideal Fertility", "Optimal IVF", "Prime Fertility", "Choice IVF", "Select Fertility", "Preferred IVF", "Chosen Fertility"]

clinic_suffixes = ["IVF Fertility", "Fertility & Birthing", "Fertility", "Infertility Centre", "ART Centre", "Fertility Clinic", "Super Speciality Hospital", "International IVF Center", "Fertility Center", "IVF Center", "Fertility Institute", "IVF Clinic", "Fertility Hospital", "IVF Hospital", "Fertility Centre", "IVF Centre", "Fertility Lab", "IVF Lab", "Fertility Solutions", "IVF Solutions", "Fertility Care", "IVF Care", "Fertility Services", "IVF Services", "Fertility Treatment", "IVF Treatment", "Fertility Therapy", "IVF Therapy", "Fertility Program", "IVF Program", "Fertility Plan", "IVF Plan", "Fertility Journey", "IVF Journey", "Fertility Path", "IVF Path", "Fertility Way", "IVF Way", "Fertility Road", "IVF Road", "Fertility Avenue", "IVF Avenue", "Fertility Street", "IVF Street", "Fertility Lane", "IVF Lane", "Fertility Boulevard", "IVF Boulevard", "Fertility Drive", "IVF Drive", "Fertility Court", "IVF Court", "Fertility Plaza", "IVF Plaza", "Fertility Square", "IVF Square", "Fertility Park", "IVF Park", "Fertility Garden", "IVF Garden", "Fertility Valley", "IVF Valley", "Fertility Hill", "IVF Hill", "Fertility Mountain", "IVF Mountain", "Fertility River", "IVF River", "Fertility Lake", "IVF Lake", "Fertility Ocean", "IVF Ocean", "Fertility Sea", "IVF Sea", "Fertility Bay", "IVF Bay", "Fertility Harbor", "IVF Harbor", "Fertility Port", "IVF Port", "Fertility Island", "IVF Island", "Fertility Peninsula", "IVF Peninsula", "Fertility Cape", "IVF Cape", "Fertility Point", "IVF Point", "Fertility Summit", "IVF Summit", "Fertility Peak", "IVF Peak", "Fertility Crest", "IVF Crest", "Fertility Ridge", "IVF Ridge", "Fertility Plateau", "IVF Plateau", "Fertility Plain", "IVF Plain", "Fertility Meadow", "IVF Meadow", "Fertility Field", "IVF Field", "Fertility Prairie", "IVF Prairie", "Fertility Savannah", "IVF Savannah", "Fertility Desert", "IVF Desert", "Fertility Oasis", "IVF Oasis", "Fertility Spring", "IVF Spring", "Fertility Well", "IVF Well", "Fertility Fountain", "IVF Fountain", "Fertility Stream", "IVF Stream", "Fertility Brook", "IVF Brook", "Fertility Creek", "IVF Creek", "Fertility River", "IVF River", "Fertility Lake", "IVF Lake", "Fertility Pond", "IVF Pond", "Fertility Pool", "IVF Pool", "Fertility Reservoir", "IVF Reservoir", "Fertility Dam", "IVF Dam", "Fertility Bridge", "IVF Bridge", "Fertility Tunnel", "IVF Tunnel", "Fertility Canal", "IVF Canal", "Fertility Aqueduct", "IVF Aqueduct", "Fertility Viaduct", "IVF Viaduct", "Fertility Causeway", "IVF Causeway", "Fertility Ferry", "IVF Ferry", "Fertility Boat", "IVF Boat", "Fertility Ship", "IVF Ship", "Fertility Yacht", "IVF Yacht", "Fertility Cruise", "IVF Cruise", "Fertility Voyage", "IVF Voyage", "Fertility Expedition", "IVF Expedition", "Fertility Adventure", "IVF Adventure", "Fertility Quest", "IVF Quest", "Fertility Mission", "IVF Mission", "Fertility Goal", "IVF Goal", "Fertility Aim", "IVF Aim", "Fertility Target", "IVF Target", "Fertility Destination", "IVF Destination", "Fertility Horizon", "IVF Horizon", "Fertility Future", "IVF Future", "Fertility Hope", "IVF Hope", "Fertility Dream", "IVF Dream"]

# Function to generate random clinic data
def generate_clinic_name():
    prefix = random.choice(clinic_prefixes)
    suffix = random.choice(clinic_suffixes)
    return f"{prefix} {suffix}"

def generate_address(city, state):
    street_types = ["Road", "Street", "Avenue", "Lane", "Nagar", "Colony", "Complex", "Center", "Hospital", "Clinic"]
    street_names = ["Main", "Central", "New", "Old", "East", "West", "North", "South", "MG", "Jawaharlal Nehru", "Gandhi", "Tagore", "Sardar Patel", "Raj", "Shivaji", "Vijay", "Ashok", "Indira", "Rajiv", "Sonia", "Modi", "Ambedkar", "Bose", "Nehru", "Gandhi Nagar", "Sector", "Phase", "Block"]
    numbers = random.randint(1, 999)
    street_name = random.choice(street_names)
    street_type = random.choice(street_types)
    return f"{numbers} {street_name} {street_type}, {city}, {state}"

def generate_phone():
    area_codes = ["011", "022", "033", "044", "080", "040", "020", "079", "0120", "0124", "0161", "0172", "0181", "0191", "0129", "0135", "0141", "0151", "0164", "0175", "0183", "0194", "0121", "0122", "0123", "0124", "0125", "0126", "0127", "0128", "0129", "0130", "0131", "0132", "0133", "0134", "0135", "0136", "0137", "0138", "0139", "0140", "0141", "0142", "0143", "0144", "0145", "0146", "0147", "0148", "0149", "0150", "0151", "0152", "0153", "0154", "0155", "0156", "0157", "0158", "0159", "0160", "0161", "0162", "0163", "0164", "0165", "0166", "0167", "0168", "0169", "0170", "0171", "0172", "0173", "0174", "0175", "0176", "0177", "0178", "0179", "0180", "0181", "0182", "0183", "0184", "0185", "0186", "0187", "0188", "0189", "0190", "0191", "0192", "0193", "0194", "0195", "0196", "0197", "0198", "0199"]
    area_code = random.choice(area_codes)
    number = random.randint(10000000, 99999999)
    return f"+91-{area_code}-{number}"

def generate_website(name):
    # Clean name for URL
    clean_name = name.lower().replace(" ", "").replace("&", "").replace(".", "").replace(",", "")
    domains = ["ivf", "fertility", "clinic", "center", "hospital", "care", "health", "medical"]
    domain = random.choice(domains)
    return f"https://www.{clean_name}{domain}.com"

def generate_description():
    descriptions = [
        "Leading fertility treatment center offering advanced ART procedures",
        "Comprehensive infertility treatment with state-of-the-art technology",
        "Specialized IVF clinic with experienced fertility specialists",
        "Advanced reproductive medicine center with high success rates",
        "Complete fertility solutions including IVF, ICSI, and fertility preservation",
        "Modern fertility clinic equipped with latest medical technology",
        "Dedicated to helping couples achieve their dream of parenthood",
        "Expert fertility care with personalized treatment plans",
        "Advanced assisted reproductive technology center",
        "Comprehensive fertility treatment and reproductive health services",
        "Specialized center for infertility diagnosis and treatment",
        "Leading IVF and fertility treatment facility",
        "Advanced fertility clinic with cutting-edge technology",
        "Complete reproductive health and fertility services",
        "Expert fertility specialists providing comprehensive care",
        "State-of-the-art fertility treatment center",
        "Advanced IVF clinic with proven success rates",
        "Comprehensive infertility treatment solutions",
        "Specialized fertility care and reproductive medicine",
        "Leading center for assisted reproductive technologies"
    ]
    return random.choice(descriptions)

# Generate 200 unique clinics
def generate_clinics_csv(filename="indian_ivf_clinics.csv", num_clinics=200):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Clinic Name', 'Address', 'City', 'State', 'Phone', 'Website', 'Description']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        generated_names = set()  # To ensure unique clinic names

        for i in range(num_clinics):
            # Select random city and state
            city, state = random.choice(cities_states)

            # Generate unique clinic name
            name = generate_clinic_name()
            while name in generated_names:
                name = generate_clinic_name()
            generated_names.add(name)

            # Generate other details
            address = generate_address(city, state)
            phone = generate_phone()
            website = generate_website(name)
            description = generate_description()

            # Write to CSV
            writer.writerow({
                'Clinic Name': name,
                'Address': address,
                'City': city,
                'State': state,
                'Phone': phone,
                'Website': website,
                'Description': description
            })

    print(f"Generated {num_clinics} unique IVF clinics in {filename}")

if __name__ == "__main__":
    generate_clinics_csv()
