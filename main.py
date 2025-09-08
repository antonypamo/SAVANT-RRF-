
#!/usr/bin/env python3
"""SAVANT-RRF demo entrypoint

Runs a small interactive demo: maps a text to a nodal index and simulates resonance.
"""
from core.engine import SavantEngine

def main():
    engine = SavantEngine()
    print("SAVANT-RRF demo. Type 'exit' to quit.")
    while True:
        q = input('You> ').strip()
        if not q: 
            continue
        if q.lower() in ('exit','quit','salir'):
            break
        out = engine.handle_query(q)
        print('SAVANT>', out)

if __name__ == '__main__':
    main()
