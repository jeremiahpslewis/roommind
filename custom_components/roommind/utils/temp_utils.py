"""Temperature unit conversion utilities for RoomMind."""

from homeassistant.const import UnitOfTemperature
from homeassistant.core import HomeAssistant


def _is_fahrenheit(hass: HomeAssistant, entity_id: str | None = None) -> bool:
    """Check if the value is in Fahrenheit.

    When *entity_id* is provided the entity's own ``unit_of_measurement``
    attribute is used.  This is race-condition-safe during HA unit-system
    changes because entity state + attributes are always updated atomically.
    Falls back to the global HA config when no entity is given.
    """
    if entity_id:
        state = hass.states.get(entity_id)
        if state:
            uom = state.attributes.get("unit_of_measurement")
            if uom:
                return bool(uom == "°F")
    return bool(hass.config.units.temperature_unit == UnitOfTemperature.FAHRENHEIT)


def ha_temp_to_celsius(
    hass: HomeAssistant,
    value: float,
    *,
    entity_id: str | None = None,
) -> float:
    """Convert temperature from HA unit system to Celsius.

    Pass *entity_id* when reading from a sensor entity to avoid race
    conditions during unit-system changes.
    """
    if _is_fahrenheit(hass, entity_id):
        return (value - 32) * 5 / 9
    return value


def celsius_to_ha_temp(hass: HomeAssistant, value: float) -> float:
    """Convert temperature from Celsius to HA unit system."""
    if hass.config.units.temperature_unit == UnitOfTemperature.FAHRENHEIT:
        return value * 9 / 5 + 32
    return value


def celsius_delta_to_ha(hass: HomeAssistant, delta: float) -> float:
    """Convert a temperature delta from Celsius to HA unit system (factor only)."""
    if hass.config.units.temperature_unit == UnitOfTemperature.FAHRENHEIT:
        return delta * 9 / 5
    return delta


def ha_delta_to_celsius(hass: HomeAssistant, delta: float) -> float:
    """Convert a temperature delta from the HA unit system to Celsius (factor only)."""
    if hass.config.units.temperature_unit == UnitOfTemperature.FAHRENHEIT:
        return delta * 5 / 9
    return delta


def ha_temp_unit_str(hass: HomeAssistant) -> str:
    """Return '°C' or '°F' based on HA config."""
    if hass.config.units.temperature_unit == UnitOfTemperature.FAHRENHEIT:
        return "°F"
    return "°C"
