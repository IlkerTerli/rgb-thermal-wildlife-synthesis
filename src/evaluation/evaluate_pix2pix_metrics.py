#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================
#  evaluate_pix2pix_metrics.py  (kommentierte Fassung)
#
#  ZWECK (B7.2): Berechnet PSNR und SSIM fuer die pix2pix-Testausgaben.
#  Es vergleicht die generierte Ausgabe (fake_B) mit dem echten
#  Thermal-Zielbild (real_B). Weil real_B bei pix2pix die PIXELGENAUE
#  Zielreferenz ist, sind PSNR und SSIM direkt anwendbar
#  -> fuer CycleGAN NICHT nutzbar (dort gibt es kein pixelgenaues real_B).
#
#  Zwei methodische Entscheidungen, die der Anhang beschreibt:
#   1. Trainings-Vorschaubilder werden ausgeschlossen (skip_path).
#   2. --common: nur Bild-IDs, die in ALLEN Laeufen vorkommen
#      -> fairer Vergleich, gleiches N (Grundlage fuer B3.7).
# =====================================================================
"""
eval_pix2pix.py -- PSNR/SSIM fuer pix2pix-Testausgaben (RGB-zu-Thermal).

Vergleicht die generierte Ausgabe (fake_B) mit dem realen Thermal-Zielbild
(real_B). Bei pix2pix ist real_B die pixelgenaue Zielreferenz, daher sind PSNR
und SSIM direkt anwendbar. NICHT fuer CycleGAN verwenden.

Eingabe je Experiment: Ordner ODER .zip. Rekursive Suche nach '*_fake_B.<ext>',
Paarung mit '*_real_B.<ext>'. '.ipynb_checkpoints' wird ignoriert.

WICHTIG -- Option --common:
    Wertet nur die Bild-IDs aus, die in ALLEN Experimenten vorkommen. Dadurch
    werden identische Szenen verglichen und N ist in allen Laeufen gleich.
    Fuer den Vergleich mehrerer Laeufe dringend empfohlen.

Aufruf (Windows, alles in EINER Zeile):
    python eval_pix2pix.py --common "Medium 20 Ep.=ex_m20" "Medium 100 Ep.=ex_m100" ...

Weitere Optionen:
    --color gray|rgb   (Standard: gray)
    --out-prefix NAME  (Standard: metrics)
    --limit N          (nur erste N Tripel je Experiment; 0 = alle)

Abhaengigkeiten: numpy, pillow, scikit-image
"""

import argparse   # Kommandozeilen-Argumente einlesen (Experimente, --common, --color ...)
import csv        # Ergebnisse als CSV-Dateien schreiben
import io          # Bilder aus ZIP-Archiven im Speicher oeffnen (BytesIO)
import os          # Pfade, Dateisystem, Verzeichnisse durchlaufen
import sys         # Fehlerausgaben (stderr) und Programmabbruch (exit)
import zipfile     # Auswertung direkt aus .zip-Archiven ermoeglichen

import numpy as np           # numerische Bild-Arrays und Statistik (Mittelwert/Streuung)
from PIL import Image        # Bilder laden und in Graustufen/RGB konvertieren

# scikit-image ist die Standardbibliothek fuer PSNR/SSIM. Falls sie fehlt,
# wird ein eigener numpy-Fallback verwendet (weiter unten definiert).
try:
    from skimage.metrics import structural_similarity as sk_ssim   # SSIM (Wang et al. 2004)
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr # PSNR
    HAVE_SKIMAGE = True     # scikit-image ist vorhanden -> bevorzugt nutzen
except Exception:
    HAVE_SKIMAGE = False    # nicht vorhanden -> numpy-Fallback

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")   # erlaubte Bild-Endungen
SKIP_DIR = ".ipynb_checkpoints"   # Jupyter-Hilfsordner, wird ignoriert
FAKE = "_fake_b"                  # Namensbaustein, an dem eine generierte Ausgabe erkannt wird

# ENTSCHEIDUNG 1: Trainings-Vorschaubilder, die pix2pix waehrend des Trainings unter
# checkpoints/<name>/web/images ablegt, sind KEINE Testausgaben und werden
# standardmaessig ignoriert. So geraten keine Trainingsdaten in die Testauswertung.
SKIP_TRAIN_WEB = True


def skip_path(path):
    # Entscheidet, ob ein Pfad von der Auswertung AUSGESCHLOSSEN wird.
    q = path.replace("\\", "/").lower()   # Pfad vereinheitlichen (Windows-/Unix-Trenner, Kleinschreibung)
    if SKIP_DIR in q:                     # Jupyter-Checkpoints?
        return True                       # -> ausschliessen
    # Trainings-Vorschaubilder aus dem checkpoints-/web-Verzeichnis ausschliessen:
    if SKIP_TRAIN_WEB and ("/web/images" in q or q.endswith("/web")
                           or "/checkpoints/" in q):
        return True                       # -> ausschliessen (nur echte Testausgaben zaehlen)
    return False                          # sonst: Pfad wird ausgewertet


# ----------------------------------------------------------------- Hilfsroutinen
def to_array(img, color):
    # Wandelt ein Bild in ein Zahlen-Array um -- standardmaessig in GRAUSTUFEN (B7.3).
    # Grund: thermalaehnliche Bilder sind einkanalige Intensitaetsbilder; bewertet wird
    # die Helligkeits-/Strukturinformation, nicht Farbkanaele.
    img = img.convert("L") if color == "gray" else img.convert("RGB")  # "L" = Graustufe
    return np.asarray(img, dtype=np.uint8)   # als uint8-Array (Werte 0..255)


def match_size(fake, real):
    # Sicherheitsschritt: Falls fake_B und real_B nicht exakt gleich gross sind,
    # wird fake_B auf die Masse von real_B skaliert (PSNR/SSIM brauchen gleiche Dimensionen).
    if fake.shape[:2] == real.shape[:2]:   # bereits gleich gross?
        return fake, False                 # -> unveraendert, "wurde nicht skaliert"
    mode = "L" if fake.ndim == 2 else "RGB"   # Graustufe oder Farbe?
    r = Image.fromarray(fake, mode=mode).resize((real.shape[1], real.shape[0]),
                                                Image.BILINEAR)   # auf real-Groesse skalieren
    return np.asarray(r, dtype=np.uint8), True   # skaliertes Array + Kennzeichen "wurde skaliert"


def is_fake(nl):
    # Prueft, ob ein Dateiname eine generierte Ausgabe ist (endet auf _fake_b.<endung>).
    return any(nl.endswith(FAKE + e) for e in IMG_EXTS)


def pair_id(fname):
    """'pair_03376_fake_B.png' -> 'pair_03376'"""
    # Extrahiert die gemeinsame Bild-ID, indem der _fake_b-Teil abgeschnitten wird.
    # Ueber diese ID werden fake_B und real_B (und Laeufe untereinander) einander zugeordnet.
    nl = fname.lower()
    i = nl.rfind(FAKE)                  # Position des _fake_b-Bausteins
    return fname[:i] if i != -1 else fname   # alles davor = die Bild-ID


def real_candidates(fname):
    # Erzeugt zu einem fake_B-Dateinamen die moeglichen real_B-Namen
    # (durch Ersetzen von _fake_B/_fake_b durch _real_B/_real_b).
    out = []
    for a, b in (("_fake_B", "_real_B"), ("_fake_b", "_real_b")):   # Gross-/Kleinschreibung abdecken
        if a in fname:
            out.append(fname.replace(a, b))
    nl = fname.lower()
    i = nl.rfind(FAKE)
    if i != -1:                         # zusaetzliche Varianten robust erzeugen
        out.append(fname[:i] + "_real_B" + fname[i + len(FAKE):])
        out.append(fname[:i] + "_real_b" + fname[i + len(FAKE):])
    seen, res = set(), []               # Duplikate herausfiltern
    for c in out:
        if c != fname and c not in seen:
            seen.add(c); res.append(c)
    return res                          # Liste moeglicher real_B-Namen


# ----------------------------------------------------------------- Metriken
def psnr(a, b):
    # PSNR (Peak Signal-to-Noise Ratio): hoeher = aehnlicher. Logarithmische dB-Skala.
    if HAVE_SKIMAGE:
        return float(sk_psnr(b, a, data_range=255))   # scikit-image, Wertebereich 0..255
    # numpy-Fallback (falls scikit-image fehlt): PSNR direkt aus dem mittleren Fehler (MSE)
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def _gauss(size=11, sigma=1.5):
    # Gauss-Fenster fuer den SSIM-Fallback (gewichtete lokale Nachbarschaft).
    ax = np.arange(size) - (size - 1) / 2.0
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    return g / g.sum()                  # normiert auf Summe 1


def _ssim_np(a, b):
    # numpy-Fallback fuer SSIM (nur genutzt, wenn scikit-image fehlt).
    from scipy.ndimage import convolve1d
    a, b = a.astype(np.float64), b.astype(np.float64)
    k = _gauss(); C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2   # Stabilisierungskonstanten
    blur = lambda x: convolve1d(convolve1d(x, k, 0, mode="reflect"), k, 1, mode="reflect")
    ma, mb = blur(a), blur(b)                          # lokale Mittelwerte
    sa, sb = blur(a * a) - ma ** 2, blur(b * b) - mb ** 2   # lokale Varianzen
    sab = blur(a * b) - ma * mb                        # lokale Kovarianz
    m = ((2 * ma * mb + C1) * (2 * sab + C2)) / ((ma ** 2 + mb ** 2 + C1) * (sa + sb + C2))
    return float(m.mean())              # mittlere lokale Strukturaehnlichkeit


def ssim(a, b, color):
    # SSIM (Structural Similarity): misst lokale Struktur/Kontrast/Luminanz, 0..1, hoeher = aehnlicher.
    if HAVE_SKIMAGE:
        ch = None if color == "gray" else -1            # Graustufe: kein Farbkanal
        ms = min(a.shape[0], a.shape[1])                # kleinste Bildkante
        win = max(3, min(7, ms if ms % 2 else ms - 1))  # Fensterbreite: ungerade und <= Bildkante (Robustheit bei kleinen Bildern)
        return float(sk_ssim(b, a, data_range=255, channel_axis=ch, win_size=win))
    if color != "gray":                                 # Fallback fuer Farbe: pro Kanal mitteln
        return float(np.mean([_ssim_np(a[..., c], b[..., c]) for c in range(3)]))
    return _ssim_np(a, b)                               # Fallback fuer Graustufe


# ----------------------------------------------------------------- Index bauen
def index_zip(zip_path):
    # Baut aus einem .zip-Archiv den Index: Bild-ID -> (Typ, zip, fake-Pfad, real-Pfad).
    idx, dirs = {}, set()
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist()
                   if not m.endswith("/") and not skip_path(m)]   # Ordner + ausgeschlossene Pfade weglassen
        mset = set(members)
        for fm in sorted(members):
            base = os.path.basename(fm)
            if not is_fake(base.lower()):     # nur fake_B-Dateien als Startpunkt
                continue
            d = os.path.dirname(fm)
            # passenden real_B-Namen im selben Verzeichnis suchen:
            rm = next((os.path.join(d, c).replace("\\", "/")
                       for c in real_candidates(base)
                       if os.path.join(d, c).replace("\\", "/") in mset), None)
            if rm is None:                    # kein real_B gefunden?
                continue                      # -> ueberspringen
            pid = pair_id(base)               # gemeinsame Bild-ID
            dirs.add(d)
            idx.setdefault(pid, ("zip", zip_path, fm, rm))   # Eintrag pro Bild-ID
    return idx, dirs


def index_dir(root):
    # Wie index_zip, aber fuer einen normalen Ordner (rekursiv durchlaufen).
    idx, dirs = {}, set()
    for dirpath, _, files in os.walk(root):   # alle Unterordner durchgehen
        if skip_path(dirpath):                # ausgeschlossene Ordner (web/images, checkpoints) ueberspringen
            continue
        fset = set(files)
        for f in sorted(files):
            if not is_fake(f.lower()):        # nur fake_B-Dateien
                continue
            rf = next((c for c in real_candidates(f) if c in fset), None)  # passendes real_B suchen
            if rf is None:
                continue
            pid = pair_id(f)
            dirs.add(dirpath)
            idx.setdefault(pid, ("dir", None,
                                 os.path.join(dirpath, f), os.path.join(dirpath, rf)))
    return idx, dirs


def build_index(root):
    # Waehlt je nach Eingabe (Ordner oder .zip) die passende Index-Funktion.
    if os.path.isfile(root) and root.lower().endswith(".zip"):
        return index_zip(root)
    if os.path.isdir(root):
        return index_dir(root)
    print(f"  [WARN] Weder Ordner noch .zip: {root}", file=sys.stderr)
    return {}, set()


def load_pair(entry, color):
    # Laedt ein fake_B/real_B-Paar (aus Ordner oder aus dem ZIP) als Arrays.
    kind, zpath, fpath, rpath = entry
    if kind == "zip":
        with zipfile.ZipFile(zpath) as zf:
            fake = to_array(Image.open(io.BytesIO(zf.read(fpath))), color)
            real = to_array(Image.open(io.BytesIO(zf.read(rpath))), color)
    else:
        fake = to_array(Image.open(fpath), color)
        real = to_array(Image.open(rpath), color)
    return fake, real                     # (generierte Ausgabe, echtes Zielbild)


# ----------------------------------------------------------------- Auswertung
def evaluate(label, idx, ids, color, writer):
    # Berechnet fuer EIN Experiment ueber alle Bild-IDs die Metriken und mittelt sie.
    pv, sv, resized = [], [], 0           # PSNR-Werte, SSIM-Werte, Zaehler skalierter Bilder
    for pid in ids:
        fake, real = load_pair(idx[pid], color)   # Paar laden
        fake, was = match_size(fake, real)        # ggf. auf gleiche Groesse bringen
        resized += int(was)                       # mitzaehlen, falls skaliert
        p, s = psnr(fake, real), ssim(fake, real, color)   # Metriken berechnen
        pv.append(p); sv.append(s)
        writer.writerow([label, pid, f"{p:.4f}", f"{s:.4f}"])   # Einzelwert je Bild in CSV schreiben
    if not pv:
        return None                       # nichts ausgewertet
    pv, sv = np.array(pv, float), np.array(sv, float)
    fin = np.isfinite(pv)                 # unendliche PSNR-Werte (identische Bilder) herausfiltern
    # Rueckgabe: Mittelwert UND Standardabweichung -> die "+/-" Werte in Tabelle 6.2
    return {"label": label, "n": len(pv),
            "psnr_mean": float(pv[fin].mean()) if fin.any() else float("inf"),
            "psnr_std": float(pv[fin].std()) if fin.any() else 0.0,
            "ssim_mean": float(sv.mean()), "ssim_std": float(sv.std()),
            "ssim_min": float(sv.min()), "ssim_max": float(sv.max()),
            "resized": resized}


def print_console(res, common):
    # Gibt die Ergebnistabelle lesbar in der Konsole aus.
    print("\n" + "=" * 74)
    print(f"{'Experiment':<22}{'N':>5}{'PSNR (dB)':>16}{'SSIM':>17}")
    print("-" * 74)
    for r in res:
        print(f"{r['label']:<22}{r['n']:>5}"
              f"{r['psnr_mean']:>9.2f} +/-{r['psnr_std']:<4.2f}"
              f"{r['ssim_mean']:>9.3f} +/-{r['ssim_std']:<5.3f}")
    print("=" * 74)
    if common:
        print("Modus: --common (identische Bild-IDs in allen Experimenten)")


def print_latex(res, color, common):
    # Erzeugt die fertige LaTeX-Tabelle (im Farbstil color1/color4) fuer die Arbeit.
    dom = "Graustufen" if color == "gray" else "RGB"
    add = (" Ausgewertet wurde ausschliesslich die Schnittmenge der in allen "
           "Laeufen vorhandenen Testbeispiele." if common else "")
    print("\n% ---- LaTeX (Stil color1/color4) ----")
    print(r"\begin{table}[htbp]\begin{center}")
    print(r"\renewcommand{\arraystretch}{1.25}\setlength{\tabcolsep}{4pt}")
    print(r"\begin{tabular}{>{\raggedright\arraybackslash}p{4.2cm}"
          r">{\centering\arraybackslash}p{1.2cm}"
          r">{\centering\arraybackslash}p{3.2cm}"
          r">{\centering\arraybackslash}p{3.2cm}}")
    print(r"\rowcolor{color1}")
    print(r"\textcolor{white}{\textbf{Experiment}} & \textcolor{white}{\textbf{N}} & "
          r"\textcolor{white}{\textbf{PSNR (dB) $\uparrow$}} & "
          r"\textcolor{white}{\textbf{SSIM $\uparrow$}} \\")
    for i, r in enumerate(res):
        print(r"\rowcolor{" + ("white" if i % 2 == 0 else "color4") + "}")   # abwechselnde Zeilenfarbe
        print(f"{r['label']} & {r['n']} & "
              f"{r['psnr_mean']:.2f} $\\pm$ {r['psnr_std']:.2f} & "
              f"{r['ssim_mean']:.3f} $\\pm$ {r['ssim_std']:.3f} \\\\")
    print(r"\end{tabular}\end{center}")
    print(r"\caption{Quantitative Bildaehnlichkeit der pix2pix-Ausgaben (fake\_B) "
          r"gegenueber dem realen Thermalbild (real\_B), berechnet auf " + dom +
          r" (Mittelwert $\pm$ Standardabweichung)." + add +
          r" Hoehere Werte sind besser. Eigene Darstellung.}")
    print(r"\label{tab:pix2pix-quant}\end{table}")


def main():
    # Einstiegspunkt: Argumente lesen, Indizes bauen, IDs bestimmen, Metriken berechnen, ausgeben.
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+", help="'Label=Pfad_oder_zip'")   # ein oder mehrere Experimente
    ap.add_argument("--color", choices=["gray", "rgb"], default="gray")       # Standard: Graustufen
    ap.add_argument("--out-prefix", default="metrics")                        # Namenspraefix der CSV-Dateien
    ap.add_argument("--limit", type=int, default=0)                           # optional nur erste N Bilder
    ap.add_argument("--common", action="store_true",                          # ENTSCHEIDUNG 2: gemeinsame IDs
                    help="Nur Bild-IDs auswerten, die in ALLEN Experimenten vorkommen.")
    ap.add_argument("--all-folders", action="store_true",                     # Vorschaubilder doch einbeziehen
                    help="Auch Trainings-Vorschaubilder aus checkpoints/.../web/images "
                         "einbeziehen (Standard: ignorieren).")
    args = ap.parse_args()

    global SKIP_TRAIN_WEB
    if args.all_folders:            # nur wenn ausdruecklich gewuenscht:
        SKIP_TRAIN_WEB = False      # -> Vorschaubilder NICHT ausschliessen
    else:
        print("[Info] Trainings-Vorschaubilder unter checkpoints/.../web/images "
              "werden ignoriert (nur Testausgaben werden ausgewertet).")

    if not HAVE_SKIMAGE:
        print("[Hinweis] scikit-image fehlt -> numpy-Fallback.", file=sys.stderr)

    # 1) Indizes aufbauen -- pro Experiment: Bild-ID -> (fake_B, real_B)
    specs = []
    for spec in args.experiments:
        # Eingabeformat "Label=Pfad" zerlegen; ohne "=" wird der Ordnername als Label genutzt
        label, root = spec.split("=", 1) if "=" in spec else \
            (os.path.basename(os.path.normpath(spec)), spec)
        label, root = label.strip(), root.strip()
        idx, dirs = build_index(root)     # Index fuer dieses Experiment bauen
        print(f"[..] {label}: {root}")
        print(f"     {len(idx)} eindeutige Bild-IDs aus {len(dirs)} Quellordner(n)")
        if len(dirs) > 1:
            for d in sorted(dirs):
                print(f"       - {d}")
        if idx:
            specs.append((label, idx))    # nur nicht-leere Experimente behalten

    if not specs:
        print("Keine auswertbaren Experimente gefunden.", file=sys.stderr); sys.exit(1)

    # 2) IDs bestimmen -- HIER wirkt --common
    if args.common:
        common_ids = set(specs[0][1])     # mit den IDs des ersten Experiments starten
        for _, idx in specs[1:]:
            common_ids &= set(idx)        # SCHNITTMENGE ueber alle Laeufe -> nur gemeinsame IDs
        ids_sorted = sorted(common_ids)
        print(f"\n[--common] Schnittmenge: {len(ids_sorted)} Bild-IDs in allen "
              f"{len(specs)} Experimenten.")
        if not ids_sorted:                # keine gemeinsamen IDs (z. B. Medium vs. Large)?
            # Genau das ist die Begruendung fuer B3.7: unterschiedliche Testsplits -> nur deskriptiv vergleichbar.
            print("\nKeine gemeinsamen Bild-IDs gefunden. Ursache ist meist, dass die "
                  "verglichenen Laeufe unterschiedliche Testsplits verwenden "
                  "(z. B. Medium vs. Large).\nEmpfehlung: --common nur innerhalb "
                  "derselben Datensatzgroesse verwenden (Medium 20 vs. Medium 100, "
                  "Large 20 vs. Large 100).", file=sys.stderr)
            sys.exit(1)
        id_map = {label: ids_sorted for label, _ in specs}   # alle Laeufe nutzen dieselben IDs -> gleiches N
    else:
        id_map = {label: sorted(idx) for label, idx in specs}   # ohne --common: je Lauf alle eigenen IDs

    if args.limit:                        # optional die Menge kuenstlich begrenzen (Tests)
        id_map = {k: v[:args.limit] for k, v in id_map.items()}

    # 3) Metriken berechnen und in zwei CSV-Dateien schreiben
    per_img, summ = f"{args.out_prefix}_per_image.csv", f"{args.out_prefix}_summary.csv"
    results = []
    with open(per_img, "w", newline="", encoding="utf-8") as f:   # CSV mit Einzelwerten je Bild
        w = csv.writer(f); w.writerow(["experiment", "pair_id", "psnr", "ssim"])
        for label, idx in specs:
            r = evaluate(label, idx, id_map[label], args.color, w)   # Auswertung pro Experiment
            if r:
                results.append(r)

    with open(summ, "w", newline="", encoding="utf-8") as f:      # CSV mit Zusammenfassung (Mittelwerte)
        w = csv.writer(f)
        w.writerow(["experiment", "n", "psnr_mean", "psnr_std", "ssim_mean",
                    "ssim_std", "ssim_min", "ssim_max", "resized"])
        for r in results:
            w.writerow([r["label"], r["n"], f"{r['psnr_mean']:.4f}", f"{r['psnr_std']:.4f}",
                        f"{r['ssim_mean']:.4f}", f"{r['ssim_std']:.4f}",
                        f"{r['ssim_min']:.4f}", f"{r['ssim_max']:.4f}", r["resized"]])

    print_console(results, args.common)   # Tabelle in der Konsole
    print_latex(results, args.color, args.common)   # fertige LaTeX-Tabelle
    print(f"\nCSV: {per_img} , {summ}")


if __name__ == "__main__":   # Skript direkt gestartet -> main() ausfuehren
    main()