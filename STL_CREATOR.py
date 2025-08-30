import cadquery as cq

# --- Input utente ---
raggio = float(input("Inserisci il raggio del profilo (mm): "))
asse = input("Inserisci l'asse di rivoluzione (x, y, z): ").lower().strip()
# Distanza del centro dello sketch dall'asse di rivoluzione (necessaria per evitare autointersezioni)
# Se vuoi un toro con centro sull'asse (NON valido), imposta 0; consigliato: usare un valore > 0
offset = float(input("Distanza del centro profilo dall'asse (mm): "))

# --- Scelta del workplane e dell'asse (l'asse DEVE giacere nel piano dello sketch) ---
# Mappiamo l'asse richiesto su un workplane che lo contenga e definiamo i punti axisStart/axisEnd
if asse == "x":
    # L'asse X giace nei piani XY e XZ; usiamo XZ
    wp = cq.Workplane("XZ")
    axis_start = (0, 0, 0)
    axis_end = (1, 0, 0)  # direzione X
    # Nel piano XZ, l'asse X è la riga Z=0; spostiamo il centro lungo Z
    move_to = (0, offset)  # (X,Z)
elif asse == "y":
    # L'asse Y giace nei piani YZ e XY; usiamo YZ
    wp = cq.Workplane("YZ")
    axis_start = (0, 0, 0)
    axis_end = (0, 1, 0)  # direzione Y
    # Nel piano YZ, l'asse Y è la riga Z=0; spostiamo il centro lungo Z
    move_to = (0, offset)  # (Y,Z)
else:
    # default: asse Z, che giace nei piani XZ e YZ; usiamo XZ
    wp = cq.Workplane("XZ")
    axis_start = (0, 0, 0)
    axis_end = (0, 0, 1)  # direzione Z
    # Nel piano XZ, l'asse Z è la riga X=0; spostiamo il centro lungo X
    move_to = (offset, 0)  # (X,Z)

# --- Costruzione profilo ---
# Usiamo un cerchio come profilo chiuso: ATTENZIONE
# Per una rivoluzione valida, il centro del cerchio non deve coincidere con l'asse (offset>0),
# altrimenti si crea un toro auto-intersecante o un'operazione degenere -> errore BRep_API.
profilo = wp.moveTo(*move_to).circle(raggio)

# --- Rivoluzione ---
try:
    solido = profilo.revolve(angleDegrees=360, axisStart=axis_start, axisEnd=axis_end)
except Exception as e:
    raise RuntimeError(
        "Rivoluzione fallita. Verifica che l'offset sia > 0 e che il profilo non intersechi l'asse."
    ) from e

# --- Esporta ---
cq.exporters.export(solido, "output.step")
cq.exporters.export(solido, "output.stl")
print("Oggetto creato: output.step e output.stl")