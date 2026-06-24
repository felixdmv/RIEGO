import numpy as np

# Known layout: row_spacing = 6.0m, dripper_spacing = 0.75m, dripper_flow = 1.6 l/h, 2 lines per row
row_spacing = 6.0
dripper_spacing = 0.75
dripper_flow = 1.6  # l/h
lines_per_row = 2

def estimate_area_from_flow(flow_m3_h):
    flow_l_h = flow_m3_h * 1000.0
    num_emitters = flow_l_h / dripper_flow
    dripline_len = num_emitters * dripper_spacing
    row_len = dripline_len / lines_per_row
    area_m2 = row_len * row_spacing
    return area_m2

# Calculated median flow rates:
# V1: 24.917 m3/h
# V2: 18.212 m3/h
# V3: 29.517 m3/h (Actual area is 40650 m2, which is 4.07 ha)
# V4: 28.308 m3/h

print("--- ESTIMATING SECTOR AREAS FROM MEDIAN FLOW RATES ---")
for valve, flow in [('V1', 24.917), ('V2', 18.212), ('V3', 29.517), ('V4', 28.308)]:
    est_area = estimate_area_from_flow(flow)
    print(f"Valve {valve}:")
    print(f"  Flow Rate: {flow:.3f} m3/h")
    print(f"  Estimated Area: {est_area:.1f} m2 ({est_area/10000.0:.3f} ha)")
    if valve == 'V3':
        print(f"    Difference from actual (40650 m2): {est_area - 40650:.1f} m2 ({((est_area - 40650)/40650)*100:+.2f}%)")
