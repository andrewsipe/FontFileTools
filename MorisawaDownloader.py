import urllib.request
import re
import os
from urllib.parse import urljoin, urlparse

# --- Configuration ---
# Enter your URLs separated by commas
INPUT_URLS = "https://stg.morisawafonts.net/j/01K317BK34K932KRQ29PWQNKMZ/0v380grfk5.css, https://stg.morisawafonts.net/j/01K317BK34K932KRQ29PWQNKMZ/2cqgy8xq1w.css, https://stg.morisawafonts.net/j/01K317BK34K932KRQ29PWQNKMZ/2ofalbih28.css, https://stg.morisawafonts.net/j/01K317BK34K932KRQ29PWQNKMZ/4c31ktf5el.css, https://stg.morisawafonts.net/j/01K317BK34K932KRQ29PWQNKMZ/5ubjdawu53.css, https://stg.morisawafonts.net/j/01K317BK34K932KRQ29PWQNKMZ/6nkov5ok3o.css, https://stg.morisawafonts.net/j/01K317BK34K932KRQ29PWQNKMZ/6q6ystrnuc.css, https://stg.morisawafonts.net/j/01K317BK34K932KRQ29PWQNKMZ/8ayp92md3p.css, https://stg.morisawafonts.net/j/01K317BK34K932KRQ29PWQNKMZ/8jymqmi7ht.css, https://stg.morisawafonts.net/j/01K317BK34K932KRQ29PWQNKMZ/41g2meqaar.css, https://stg.morisawafonts.net/j/01K317BK34K932KRQ29PWQNKMZ/47qcud0vwb.css, https://stg.morisawafonts.net/j/01K317BK34K932KRQ29PWQNKMZ/95cj9shvkx.css, https://stg.morisawafonts.net/j/01K317BK34K932KRQ29PWQNKMZ/ahe6d352u1.css, https://stg.morisawafonts.net/j/01K317BK34K932KRQ29PWQNKMZ/c9txipjmbo.css, https://stg.morisawafonts.net/j/01K317BK34K932KRQ29PWQNKMZ/f3dsfnw6kb.css, https://stg.morisawafonts.net/j/01K317BK34K932KRQ29PWQNKMZ/hxso0ac7kq.css, https://stg.morisawafonts.net/j/01K317BK34K932KRQ29PWQNKMZ/kctgc1c0ny.css, https://stg.morisawafonts.net/j/01K317BK34K932KRQ29PWQNKMZ/lv195l9537.css, https://stg.morisawafonts.net/j/01K317BK34K932KRQ29PWQNKMZ/mark4vqc0p.css, https://stg.morisawafonts.net/j/01K317BK34K932KRQ29PWQNKMZ/mjliip4kwt.css, https://stg.morisawafonts.net/j/01K317BK34K932KRQ29PWQNKMZ/ocbu3n305y.css, https://stg.morisawafonts.net/j/01K317BK34K932KRQ29PWQNKMZ/owf6l212si.css, https://stg.morisawafonts.net/j/01K317BK34K932KRQ29PWQNKMZ/r2ybkx8qgi.css, https://stg.morisawafonts.net/j/01K317BK34K932KRQ29PWQNKMZ/r7v32q7ipv.css, https://stg.morisawafonts.net/j/01K317BK34K932KRQ29PWQNKMZ/vl7be5yxe6.css, https://stg.morisawafonts.net/j/01K317BK34K932KRQ29PWQNKMZ/ya2b901dpm.css, https://stg.morisawafonts.net/j/01K317BK34K932KRQ29PWQNKMZ/ydowd91vhk.css, https://stg.morisawafonts.net/j/01K317BK34K932KRQ29PWQNKMZ/zfk9g3364k.css, https://stg.morisawafonts.net/j/01K317BK34K932KRQ29PWQNKMZ/zici0xyyai.css, https://stg.morisawafonts.net/j/01K317BK34K932KRQ29PWQNKMZ/zpfdh12lbv.css"
BASE_URL = "https://stg.morisawafonts.net"
OUTPUT_DIR = "latin_fonts_batch"
# ---------------------

def download_font(url, filename, referer):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Referer': referer
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response, open(filename, 'wb') as f:
            f.write(response.read())
        return True
    except Exception as e:
        print(f"  -> Error downloading {url}: {e}")
        return False

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Split the input string by commas and clean up whitespace
    urls = [u.strip() for u in INPUT_URLS.split(",") if u.strip()]

    total_downloaded = 0

    for css_url in urls:
        print(f"\nProcessing CSS: {css_url}")

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        try:
            req = urllib.request.Request(css_url, headers=headers)
            with urllib.request.urlopen(req) as response:
                css_content = response.read().decode('utf-8')
        except Exception as e:
            print(f"Failed to fetch CSS {css_url}: {e}")
            continue

        blocks = re.findall(r'@font-face\s*\{([^}]+)\}', css_content)

        for block in blocks:
            # Skip blocks with unicode-range (typically Japanese subsets)
            if "unicode-range" in block.lower():
                continue

            family_match = re.search(r'font-family\s*:\s*[\'"]?([^\'";]+)[\'"]?', block)
            url_match = re.search(r'url\(\s*[\'"]?([^\'")]+)[\'"]?\s*\)', block)

            if family_match and url_match:
                font_family = family_match.group(1).strip()
                font_url_partial = url_match.group(1).strip()

                full_url = urljoin(BASE_URL, font_url_partial)
                ext = os.path.splitext(urlparse(full_url).path)[1]

                # Create a safe filename
                safe_name = re.sub(r'[^A-Za-z0-9_\-]', '', font_family.replace(' ', '-'))
                filename = f"{safe_name}{ext}"
                output_path = os.path.join(OUTPUT_DIR, filename)

                # Handle duplicate filenames across different CSS files
                counter = 1
                while os.path.exists(output_path):
                    output_path = os.path.join(OUTPUT_DIR, f"{safe_name}_{counter}{ext}")
                    counter += 1

                print(f" -> Downloading: {os.path.basename(output_path)}")
                if download_font(full_url, output_path, css_url):
                    total_downloaded += 1

    print(f"\nTask Complete! Total files saved to '{OUTPUT_DIR}': {total_downloaded}")

if __name__ == "__main__":
    main()
