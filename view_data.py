from main import app
from models import User, Clinic, PatientData

def print_users():
    """Prints all users in the database."""
    print("\n--- Users Table ---")
    users = User.query.all()
    if not users:
        print("No users found.")
        return
    for user in users:
        print(
            f"ID: {user.id:<3} | "
            f"Username: {user.username:<15} | "
            f"Email: {user.email:<25} | "
            f"Type: {user.user_type:<8} | "
            f"Clinic ID: {user.clinic_id or 'N/A'}"
        )
    print("-" * 80)

def print_clinics():
    """Prints all clinics in the database."""
    print("\n--- Clinics Table ---")
    clinics = Clinic.query.all()
    if not clinics:
        print("No clinics found.")
        return
    for clinic in clinics:
        print(
            f"ID: {clinic.id:<3} | "
            f"Name: {clinic.name:<25} | "
            f"Location: {clinic.city}, {clinic.state}"
        )
    print("-" * 80)

if __name__ == "__main__":
    with app.app_context():
        print("Querying database to view data...")
        print_users()
        print_clinics()
        print("\nData view complete.")