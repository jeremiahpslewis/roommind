"""Service for updating RoomMind global settings from automations/dev tools.

Exists primarily for settings that have no UI field yet (e.g. the whole-house
ventilation supply sensor). The store merges changes, so keys set here survive
later UI saves.
"""

from __future__ import annotations

import logging

import voluptuous as vol
from homeassistant.core import HomeAssistant, ServiceCall

from ..const import DOMAIN

_LOGGER = logging.getLogger(__name__)

SERVICE_UPDATE_SETTINGS = "update_settings"

# Keys settable through the service. Deliberately a whitelist: arbitrary keys
# could shadow structured settings the UI owns.
_ALLOWED_KEYS = {
    "ventilation_supply_sensor",
}

_SCHEMA = vol.Schema(
    {vol.Required("settings"): vol.Schema({vol.In(_ALLOWED_KEYS): vol.Any(str, None)})},
)


def async_register_settings_service(hass: HomeAssistant) -> None:
    """Register the roommind.update_settings service."""

    async def _handle(call: ServiceCall) -> None:
        store = hass.data.get(DOMAIN, {}).get("store")
        if store is None:
            _LOGGER.warning("roommind.update_settings called before the store is ready")
            return
        changes = dict(call.data["settings"])
        await store.async_save_settings(changes)
        _LOGGER.info("RoomMind settings updated via service: %s", changes)

    hass.services.async_register(DOMAIN, SERVICE_UPDATE_SETTINGS, _handle, schema=_SCHEMA)
