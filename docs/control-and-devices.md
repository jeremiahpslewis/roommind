# Control And Device Guide

This page explains RoomMind's control settings and related device options.

## What Priority Does

In `Settings -> Control -> Priority`, the slider balances comfort against runtime/energy use for MPC.

- Toward `Comfort`: RoomMind reacts earlier and works harder to stay close to the target temperature.
- Toward `Efficiency`: RoomMind allows more drift around the target to reduce heating/cooling runtime.

This setting does **not** change your schedule targets, overrides, comfort temperature, or eco temperature. It only changes how aggressively MPC tries to reach and hold those targets.

## How Far AC Setpoints Are Pushed

For climate devices RoomMind sends a setpoint past the room target to shape output (see `Proportional` below). An AC's compressor speed and fan speed both rise with the gap between the room and the setpoint it was given — a unit told `19°C` in a `22°C` room does not ease toward it, it runs hard and loud. The same command is also the temperature the unit will eventually drive the room to.

So the excursion past target is kept proportional to the error RoomMind is actually correcting, and the `Priority` slider sets how aggressive that proportion is:

- toward `Efficiency`: a small excursion, so the unit runs quietly and takes longer
- toward `Comfort`: a larger excursion, so the unit works harder and corrects sooner

Only error in the direction being corrected counts — a room that has already overshot past target gets the minimum excursion, not the largest.

For a room `0.1°C` above a `22°C` cooling target, the device is given roughly `21.7°C` at full `Efficiency` and `21.2°C` at full `Comfort`, rather than being sent to its minimum. A room several degrees above target still gets the full pull-down setpoint.

On a device that only accepts whole-degree setpoints the excursion is rounded up to one whole step, because a fraction of a degree would round back onto the target and leave the unit with no demand at all.

### Learned Gap Response (beta)

The rules above are a heuristic: they assume how much cooling a given gap buys. RoomMind can instead **measure** it, per device.

Your AC reports two things HA can read — its own sensor (`current_temperature`) and the setpoint it is regulating against. The difference is the gap actually driving its compressor and fan. RoomMind compares the room's temperature change against the passive drift its thermal model predicts, and the residual is the work the unit did. That gives a curve of gap → °C/h, plus the `T_head − T_room` offset, which is why a "boost" was ever needed: the unit stops when *its* sensor reads the setpoint, not yours.

Once identified, commanding becomes a measurement rather than a guess:

```
setpoint = (room temperature + offset) − gap_that_delivers_the_required_rate
```

Because the learned curve saturates, it also knows the point past which more gap buys noise instead of cooling — something no fixed rule can know per device.

This is **observation-only until it has enough data**: it needs a spread of different gaps, not just many samples at one, so it activates after the room has been through some varied conditions. Until then the heuristic above stays in charge. Cooling only for now; heating is learned but still commanded by the heuristic.

To inspect it, download diagnostics (`Settings → Devices & Services → RoomMind → ⋮ → Download diagnostics`) and look for `gap_response` under the room. `driving_setpoints` tells you whether the curve is in charge yet, `gap_spread_K` how much variety it has seen, and `rate_degC_per_h` is the curve itself. Set the `custom_components.roommind` logger to `debug` to watch each observation as it lands.

## Thermostat vs Climate Device

Both options are Home Assistant `climate.*` entities, but RoomMind treats them differently:

- `Thermostat`: a radiator thermostat / TRV style device.
- `Climate Device`: an AC, heat pump, or other climate entity used for cooling or forced-air heating.

In practice:

- Choose `Thermostat` for radiator valves and similar heating-only valve devices.
- Choose `Climate Device` for ACs, minisplits, heat pumps, and other self-contained HVAC units.

## Full Control vs Managed

An external room temperature sensor is the key split:

- `Full Control`: RoomMind uses the external sensor as the room truth and can actively shape device output.
- `Managed`: without an external room sensor, RoomMind sends target temperatures but the device mostly regulates itself using its own internal sensor.

This matters for the options below.

## Setpoint Mode: Proportional vs Direct

`Setpoint mode` is relevant for thermostat/TRV devices in `Full Control` rooms.

### Proportional

RoomMind calculates the required heating power, then sends a boosted device setpoint to achieve roughly that output.

Example:

- room target is `21°C`
- more heat is needed
- RoomMind may send `26-28°C` to the TRV to force the valve open harder

Best for:

- radiator valves / TRVs
- devices that need an exaggerated setpoint to actually deliver heat

### Direct

RoomMind sends the real target temperature and lets the device regulate itself.

Best for:

- space heaters
- pellet stoves
- devices with their own thermostat logic that should stay in control internally

## Idle Behavior: Off, Fan Only, Setback

`When idle` applies to `Climate Device` entries.

### Turn off

RoomMind turns the device off, or falls back to the device's minimum/off-like behavior if true off is not supported.

### Fan only

RoomMind keeps the device running in fan mode without active heating/cooling.

Useful when you want:

- air circulation
- less harsh on/off transitions

### Setback

RoomMind keeps the current HVAC mode active, but moves the target away from the room target:

- heating setback = `heat target - 2°C`
- cooling setback = `cool target + 2°C`

This lets the device back off instead of shutting off completely.

Important:

- the setback offset is currently fixed at `2°C`
- it is **not configurable** in the current UI

## Idle Behavior for Thermostats: Off, Low

`When idle` also applies to `Thermostat` / TRV entries, with different options.

### Turn off

RoomMind sends the TRV to its `off` state.

### Low

RoomMind keeps the TRV in its current heating mode but lowers the setpoint to the device's minimum temperature.

Useful for battery-powered Zigbee TRVs that enter deep sleep when set to `off` and then stop reacting to commands. `Low` keeps the valve responsive while effectively stopping heating.

## When "Turn off devices" Overrides `When idle`

`When idle` describes what a device should do while the room simply has no heating or cooling demand. It does **not** apply when you explicitly shut a room down via:

- `Settings → Control → Action when schedule is off` set to `Turn off devices`
- `Settings → Presence → Action when away` set to `Turn off devices`

In those cases RoomMind turns the devices off even if `When idle` is set to `Fan only` or `Setback`. Otherwise an AC would keep circulating air after the schedule ended.

The single exception is `Low` on thermostats: it stays active because the affected TRVs stop responding after being set to `off`. Lowering the setpoint to the device minimum already stops all heat output.

## Smart Source Selection

`Smart source selection` only appears when a room has:

- at least one `Thermostat` / TRV
- at least one `Climate Device` / AC
- an external temperature sensor

In that case RoomMind can decide which source should heat:

- TRV / boiler side
- AC / heat pump side
- or both, when the gap is large

It uses temperature gap and outdoor conditions to make that choice.
