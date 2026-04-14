import urllib.request
import os

def fetch_safe_data():
    print("Connecting to Project Gutenberg...")
    url = "https://www.gutenberg.org/files/11/11-0.txt" # Alice in Wonderland (Public Domain)
    save_path = "alice.txt"
    try:
        # Securely download text string safely
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = response.read()
            with open(save_path, 'wb') as f:
                f.write(data)
                
        size_kb = os.path.getsize(save_path) / 1024
        print(f"[SUCCESS] Safely downloaded testing file: alice.txt ({size_kb:.2f} KB)")
    except Exception as e:
        print("[FAIL] Could not fetch data. Error:", e)

if __name__ == "__main__":
    fetch_safe_data()
