// Keep bus activation/teardown ordering independent of GNOME UI objects.
export class ServiceLifecycle {
    constructor({start, stop, ready, error}) {
        this._start = start;
        this._stop = stop;
        this._ready = ready;
        this._error = error;
        this.enabled = false;
        this.owner = null;
        this._observedOwner = null;
        this._starting = null;
        this._stopping = Promise.resolve();
        this._restartRequested = false;
    }

    enable() {
        this.enabled = true;
        return this.ensureStarted();
    }

    disable() {
        this.enabled = false;
        if (this.owner) {
            const owner = this.owner;
            this.owner = null;
            this._queueStop(owner);
        }
    }

    observeOwner(owner) {
        const previous = this._observedOwner;
        this._observedOwner = owner;
        if (!owner && previous) {
            this.owner = null;
            if (this.enabled) {
                if (this._starting)
                    this._restartRequested = true;
                else
                    this.ensureStarted();
            }
        } else if (owner && this.enabled && this.owner !== owner && !this._starting) {
            this.owner = null;
            this.ensureStarted();
        }
    }

    _queueStop(owner) {
        this._stopping = this._stopping
            .catch(() => {})
            .then(() => this._stop(owner))
            .catch(error => this._error(error));
        return this._stopping;
    }

    ensureStarted() {
        if (!this.enabled || this.owner)
            return Promise.resolve(this.owner);
        if (this._starting)
            return this._starting;
        let failed = false;
        this._starting = (async () => {
            await this._stopping;
            if (!this.enabled)
                return null;
            const owner = await this._start();
            if (!this.enabled) {
                await this._queueStop(owner);
                return null;
            }
            if (this._restartRequested && this._observedOwner !== owner)
                return null; // The activated instance vanished before its reply.
            this.owner = owner;
            this._ready(owner);
            return owner;
        })().catch(error => {
            failed = true;
            if (this.enabled)
                this._error(error);
            return null;
        }).finally(() => {
            this._starting = null;
            const restart = this._restartRequested || !failed;
            this._restartRequested = false;
            if (restart && this.enabled && !this.owner)
                this.ensureStarted();
        });
        return this._starting;
    }
}
