# run_comp_menu.py

from aspen_comp_finder.cli import run_comp_analysis_by_address

def main():
    print("=== Aspen Comp Finder Interactive Menu ===")
    print("Type a full or partial property address to analyze comps.")
    
    address = input("📍 Enter property address: ").strip()
    if not address:
        print("❌ No address entered. Exiting.")
        return

    min_comps_input = input("🔢 Minimum comps to find (default = 3): ").strip()
    try:
        min_comps = int(min_comps_input) if min_comps_input else 3
    except ValueError:
        print("⚠️ Invalid number entered. Using default (3).")
        min_comps = 3

    print(f"\nRunning comp analysis for: {address}")
    print(f"➡️ Minimum comps: {min_comps}\n")

    try:
        run_comp_analysis_by_address(address=address, min_comps=min_comps)
    except Exception as e:
        print(f"\n🚨 Error during comp analysis: {e}")

if __name__ == "__main__":
    main()
