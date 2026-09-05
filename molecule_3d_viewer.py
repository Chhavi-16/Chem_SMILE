import json
from rdkit import Chem
from rdkit.Chem import AllChem

def get_3d_molecule_data(smiles_str):
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if not mol:
            return None
        mol_hs = Chem.AddHs(mol)
        
        res = AllChem.EmbedMolecule(mol_hs, AllChem.ETKDGv3())
        if res != 0:
            res = AllChem.EmbedMolecule(mol_hs, AllChem.ETKDGv2())
        if res != 0:
            res = AllChem.EmbedMolecule(mol_hs, useRandomCoords=True)
            
        try:
            AllChem.MMFFOptimizeMolecule(mol_hs, maxIters=300)
        except Exception:
            pass
            
        conf = mol_hs.GetConformer()
        
        atoms = []
        for i, atom in enumerate(mol_hs.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            atoms.append({
                "elem": atom.GetSymbol(),
                "x": float(pos.x),
                "y": float(pos.y),
                "z": float(pos.z)
            })
            
        bonds = []
        for bond in mol_hs.GetBonds():
            bonds.append({
                "i1": bond.GetBeginAtomIdx(),
                "i2": bond.GetEndAtomIdx(),
                "order": int(bond.GetBondTypeAsDouble())
            })
            
        if atoms:
            avg_x = sum(a["x"] for a in atoms) / len(atoms)
            avg_y = sum(a["y"] for a in atoms) / len(atoms)
            avg_z = sum(a["z"] for a in atoms) / len(atoms)
            for a in atoms:
                a["x"] -= avg_x
                a["y"] -= avg_y
                a["z"] -= avg_z
                
        return {"atoms": atoms, "bonds": bonds, "smiles": smiles_str}
    except Exception as e:
        print("3D data extraction error:", e)
        return None

def render_3d_canvas_html(smiles_str, height=280, spin=True):
    data = get_3d_molecule_data(smiles_str)
    if not data or not data["atoms"]:
        return "<div style='color:#FF1493; padding:10px;'>3D Molecule Coordinates Unavailable</div>"

    atoms_json = json.dumps(data["atoms"])
    bonds_json = json.dumps(data["bonds"])
    spin_bool = "true" if spin else "false"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background-color: #0e1117; font-family: sans-serif; }}
        #canvas_3d {{ width: 100%; height: {height}px; display: block; cursor: grab; border-radius: 8px; }}
        #canvas_3d:active {{ cursor: grabbing; }}
      </style>
    </head>
    <body>
      <canvas id="canvas_3d"></canvas>
      <script>
        (function() {{
          var atoms = {atoms_json};
          var bonds = {bonds_json};
          var spin = {spin_bool};
          var canvas = document.getElementById('canvas_3d');
          var ctx = canvas.getContext('2d');

          function resize() {{
            canvas.width = window.innerWidth || 380;
            canvas.height = {height};
          }}
          resize();
          window.addEventListener('resize', resize);

          var angleX = 0.4;
          var angleY = 0.5;
          var isDragging = false;
          var previousMousePosition = {{ x: 0, y: 0 }};

          canvas.addEventListener('mousedown', function(e) {{
            isDragging = true;
            previousMousePosition = {{ x: e.clientX, y: e.clientY }};
          }});
          canvas.addEventListener('mousemove', function(e) {{
            if (!isDragging) return;
            var deltaX = e.clientX - previousMousePosition.x;
            var deltaY = e.clientY - previousMousePosition.y;
            angleY += deltaX * 0.01;
            angleX += deltaY * 0.01;
            previousMousePosition = {{ x: e.clientX, y: e.clientY }};
          }});
          window.addEventListener('mouseup', function() {{ isDragging = false; }});

          // Touch support for mobile/tablets
          canvas.addEventListener('touchstart', function(e) {{
            if (e.touches.length === 1) {{
              isDragging = true;
              previousMousePosition = {{ x: e.touches[0].clientX, y: e.touches[0].clientY }};
            }}
          }});
          canvas.addEventListener('touchmove', function(e) {{
            if (!isDragging || e.touches.length !== 1) return;
            var deltaX = e.touches[0].clientX - previousMousePosition.x;
            var deltaY = e.touches[0].clientY - previousMousePosition.y;
            angleY += deltaX * 0.01;
            angleX += deltaY * 0.01;
            previousMousePosition = {{ x: e.touches[0].clientX, y: e.touches[0].clientY }};
          }});
          canvas.addEventListener('touchend', function() {{ isDragging = false; }});

          var colors = {{
            'C': '#505050', 'H': '#E0E0E0', 'O': '#FF3333',
            'N': '#3388FF', 'S': '#FFCC00', 'P': '#FF8800',
            'F': '#33CC33', 'CL': '#33CC33', 'BR': '#883300'
          }};
          var radii = {{
            'C': 16, 'H': 10, 'O': 15, 'N': 15, 'S': 18, 'P': 18, 'F': 14, 'CL': 16, 'BR': 18
          }};

          function draw() {{
            ctx.fillStyle = '#0e1117';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            var scale = Math.min(canvas.width, canvas.height) / 7.5;
            var cx = canvas.width / 2;
            var cy = canvas.height / 2;

            var proj = atoms.map(function(a) {{
              var x = a.x, y = a.y, z = a.z;
              var x1 = x * Math.cos(angleY) + z * Math.sin(angleY);
              var z1 = -x * Math.sin(angleY) + z * Math.cos(angleY);
              var y2 = y * Math.cos(angleX) - z1 * Math.sin(angleX);
              var z2 = y * Math.sin(angleX) + z1 * Math.cos(angleX);

              return {{
                elem: a.elem,
                px: cx + x1 * scale,
                py: cy + y2 * scale,
                pz: z2,
                r: (radii[a.elem.toUpperCase()] || 14) * (1 + z2 * 0.06)
              }};
            }});

            var bondList = bonds.map(function(b) {{
              var a1 = proj[b.i1];
              var a2 = proj[b.i2];
              return {{ a1: a1, a2: a2, z: (a1.pz + a2.pz) / 2, order: b.order }};
            }});
            bondList.sort(function(a, b) {{ return a.z - b.z; }});

            bondList.forEach(function(b) {{
              ctx.beginPath();
              ctx.moveTo(b.a1.px, b.a1.py);
              ctx.lineTo(b.a2.px, b.a2.py);
              ctx.strokeStyle = 'rgba(180, 180, 180, 0.8)';
              ctx.lineWidth = b.order === 2 ? 5 : b.order === 3 ? 7 : 3;
              ctx.stroke();
            }});

            var sortedAtoms = proj.slice().sort(function(a, b) {{ return a.pz - b.pz; }});
            sortedAtoms.forEach(function(a) {{
              var r = Math.max(4, a.r);
              ctx.beginPath();
              ctx.arc(a.px, a.py, r, 0, 2 * Math.PI);
              var col = colors[a.elem.toUpperCase()] || '#AAAAAA';
              
              var grad = ctx.createRadialGradient(
                a.px - r * 0.35, a.py - r * 0.35, r * 0.1,
                a.px, a.py, r
              );
              grad.addColorStop(0, '#FFFFFF');
              grad.addColorStop(0.3, col);
              grad.addColorStop(1, '#111111');
              
              ctx.fillStyle = grad;
              ctx.fill();
              ctx.strokeStyle = 'rgba(0,0,0,0.6)';
              ctx.lineWidth = 1;
              ctx.stroke();
            }});

            if (spin && !isDragging) {{
              angleY += 0.012;
            }}
            requestAnimationFrame(draw);
          }}

          draw();
        }})();
      </script>
    </body>
    </html>
    """
    return html
