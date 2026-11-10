# Derived recovery verification v2

The first skidpad collection finished with zero physical contacts and zero off-course events. Original verification stopped at environment comparison because JSON stores VehicleParams.v_blend as a list and dataclasses.asdict retains its tuple. The camera, mount, scene and all vehicle scalar values matched. The original failed receipt and original collector, verifier, protocol and source archive remain unchanged.

verify_v2.py compares the expected vehicle dataclass after the same JSON serialization used by the producer. It does not alter numerical tolerances, scene parameters, arrays, model outputs, teacher targets or any physical qualification gate. It binds the original archived verifier and its own derived source hash separately. The original 142-file producer inventory must still match byte for byte. A new external receipt records the derived audit. Successful collection is data evidence only, not learned driving performance.
