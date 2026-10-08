"""Cálculo anticipado en un hilo: decodificar el siguiente frame de video mientras se dibuja el actual."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Generic, Hashable, Optional, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class Lookahead(Generic[K, V]):
    """Mantiene como mucho un cálculo `fn(key)` en marcha en un hilo auxiliar.

    `get(key)` devuelve el resultado anticipado si coincide la clave (esperando a que termine si hace falta)
    y, si no, calcula `fn(key)` en el hilo que llama. `schedule(key)` lanza el cálculo del siguiente valor.
    El hilo auxiliar y el principal nunca ejecutan `fn` a la vez: `get` siempre espera al cálculo pendiente
    antes de calcular por su cuenta, así `fn` puede usar recursos no reentrantes (un VideoCapture).
    """

    def __init__(self, fn: Callable[[K], V]):
        self.fn = fn
        self._pool: Optional[ThreadPoolExecutor] = None
        self._pending: Optional[tuple[K, Future]] = None
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _executor(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lookahead")
        return self._pool

    def get(self, key: K) -> V:
        with self._lock:
            pending, self._pending = self._pending, None
        if pending is not None:
            pkey, fut = pending
            try:
                value = fut.result()
            except Exception:  # noqa: BLE001 - el fallo se repetirá (y se verá) al calcular aquí
                value = None
                pkey = None
            if pkey == key:
                self.hits += 1
                return value  # type: ignore[return-value]
        self.misses += 1
        return self.fn(key)

    def schedule(self, key: K) -> None:
        with self._lock:
            old = self._pending
            if old is not None and old[0] == key:
                return
            self._pending = None
        if old is not None:  # nunca dos cálculos a la vez: esperar al anterior antes de lanzar otro
            try:
                old[1].result()
            except Exception:  # noqa: BLE001
                pass
        with self._lock:
            self._pending = (key, self._executor().submit(self.fn, key))

    def close(self) -> None:
        with self._lock:
            pending, self._pending = self._pending, None
            pool, self._pool = self._pool, None
        if pending is not None:
            try:
                pending[1].result()
            except Exception:  # noqa: BLE001
                pass
        if pool is not None:
            pool.shutdown(wait=True)
