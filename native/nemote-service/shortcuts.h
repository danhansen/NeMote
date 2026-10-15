#pragma once
#include <gio/gio.h>
#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// The desktop owns shortcut assignments and their configuration UI. The
// service owns the portal session, so closing preferences does not unbind it.
class PortalShortcuts : public std::enable_shared_from_this<PortalShortcuts> {
    static constexpr const char* destination = "org.freedesktop.portal.Desktop";
    static constexpr const char* path = "/org/freedesktop/portal/desktop";
    static constexpr const char* interface = "org.freedesktop.portal.GlobalShortcuts";
    using Reply = std::function<void(GVariant*)>;
    GDBusConnection* bus_;
    GCancellable* cancellable_ = g_cancellable_new();
    guint watch_ = 0, signals_ = 0, responses_ = 0, closed_ = 0, timeout_ = 0;
    unsigned generation_ = 0, serial_ = 0, version_ = 0;
    bool stopped_ = false, registered_ = false, busy_ = false, bound_ = false, owner_present_ = false;
    bool configure_requested_ = false;
    std::string session_, request_path_;
    Reply response_;
    std::function<void()> changed_;
    std::function<void(bool)> activation_;

    static GVariant* options(std::initializer_list<std::pair<const char*, std::string>> values = {}) {
        GVariantBuilder builder;
        g_variant_builder_init(&builder, G_VARIANT_TYPE_VARDICT);
        for (const auto& [key, value] : values)
            g_variant_builder_add(&builder, "{sv}", key, g_variant_new_string(value.c_str()));
        return g_variant_builder_end(&builder);
    }
    void notify() { if (!stopped_ && changed_) changed_(); }
    void call(const char* iface, const char* method, GVariant* args, Reply reply) {
        struct Call { std::shared_ptr<PortalShortcuts> self; unsigned generation; Reply reply; };
        g_dbus_connection_call(bus_, destination, path, iface, method, args, nullptr,
            G_DBUS_CALL_FLAGS_NONE, 10000, cancellable_,
            [](GObject* bus, GAsyncResult* result, gpointer data) {
                std::unique_ptr<Call> call(static_cast<Call*>(data));
                GError* error = nullptr;
                auto* value = g_dbus_connection_call_finish(G_DBUS_CONNECTION(bus), result, &error);
                auto& self = *call->self;
                if (!self.stopped_ && call->generation == self.generation_) {
                    if (error) self.fail(error->message);
                    else call->reply(value);
                }
                if (value) g_variant_unref(value);
                g_clear_error(&error);
            }, new Call{shared_from_this(), generation_, std::move(reply)});
    }
    void close_object(const std::string& object, const char* iface) {
        if (!object.empty())
            g_dbus_connection_call(bus_, destination, object.c_str(), iface, "Close", nullptr,
                nullptr, G_DBUS_CALL_FLAGS_NO_AUTO_START, 1000, nullptr, nullptr, nullptr);
    }
    void clear_request() {
        if (timeout_) { g_source_remove(timeout_); timeout_ = 0; }
        response_ = nullptr;
        request_path_.clear();
    }
    void reset() {
        ++generation_;
        g_cancellable_cancel(cancellable_);
        g_clear_object(&cancellable_);
        cancellable_ = g_cancellable_new();
        close_object(request_path_, "org.freedesktop.portal.Request");
        clear_request();
        auto session = std::exchange(session_, "");
        close_object(session, "org.freedesktop.portal.Session");
        busy_ = bound_ = false;
        description.clear();
        if (activation_) activation_(false); // Never leave push-to-talk held after session loss.
        hint.clear();
    }
    void fail(const std::string& message) {
        reset();
        configure_requested_ = false;
        status = "error";
        error = message;
        notify();
    }
    std::string request_token(Reply reply) {
        const auto token = "nemote_" + std::to_string(++serial_);
        std::string sender = g_dbus_connection_get_unique_name(bus_);
        sender.erase(0, 1);
        std::replace(sender.begin(), sender.end(), '.', '_');
        request_path_ = std::string(path) + "/request/" + sender + "/" + token;
        response_ = std::move(reply);
        // Subscribe before calling: a portal may emit Response before its
        // method reply reaches us. The handle token makes the path predictable.
        timeout_ = g_timeout_add_seconds(300, [](gpointer data) -> gboolean {
            auto* self = static_cast<PortalShortcuts*>(data);
            self->timeout_ = 0;
            self->fail("Shortcut setup timed out; try again in Preferences.");
            return G_SOURCE_REMOVE;
        }, this);
        return token;
    }
    void request(const char* method, GVariant* args) {
        const auto expected = request_path_;
        call(interface, method, args, [this, expected](GVariant* reply) {
            const char* handle;
            g_variant_get(reply, "(&o)", &handle);
            // The current portal API promises the token-derived path.
            if (expected != handle)
                fail("The shortcut portal returned an unexpected request handle.");
        });
    }
    void read_shortcuts(GVariant* results) {
        description.clear();
        auto* shortcuts = g_variant_lookup_value(results, "shortcuts", G_VARIANT_TYPE("a(sa{sv})"));
        if (!shortcuts) return;
        GVariantIter iterator;
        g_variant_iter_init(&iterator, shortcuts);
        const char* id;
        GVariant* properties;
        while (g_variant_iter_next(&iterator, "(&s@a{sv})", &id, &properties)) {
            const char* trigger = nullptr;
            if (std::string(id) == "dictation" && g_variant_lookup(properties, "trigger_description", "&s", &trigger))
                description = trigger;
            g_variant_unref(properties);
        }
        g_variant_unref(shortcuts);
    }
    void bind(bool edit_after_restore = false) {
        busy_ = true;
        status = "configuring";
        notify();
        auto token = request_token([this, edit_after_restore](GVariant* results) {
            read_shortcuts(results);
            bound_ = true;
            setup_completed = true;
            busy_ = false;
            configure_requested_ = false;
            status = "ready";
            notify();
            if (edit_after_restore) configure();
        });
        GVariantBuilder shortcuts;
        g_variant_builder_init(&shortcuts, G_VARIANT_TYPE("a(sa{sv})"));
        g_variant_builder_add(&shortcuts, "(s@a{sv})", "dictation",
            options({{"description", "Dictation"}}));
        request("BindShortcuts", g_variant_new("(o@a(sa{sv})s@a{sv})", session_.c_str(),
            g_variant_builder_end(&shortcuts), "", options({{"handle_token", token}})));
    }
    void create_session() {
        auto token = request_token([this](GVariant* results) {
            const char* handle = nullptr;
            if (!g_variant_lookup(results, "session_handle", "&s", &handle)) {
                fail("The shortcut portal did not return a session.");
                return;
            }
            if (!g_variant_is_object_path(handle)) { fail("Invalid shortcut session handle."); return; }
            session_ = handle;
            auto token = request_token([this](GVariant* results) {
                read_shortcuts(results);
                // GNOME may list only this session's bindings, not persisted
                // assignments. Remember successful setup and bind each fresh
                // session; GNOME owns and restores the actual key combination.
                // First-time setup still waits for an explicit Configure click.
                if (setup_completed || !description.empty() || configure_requested_)
                    bind(configure_requested_ && !description.empty());
                else { busy_ = false; status = "not-configured"; notify(); }
            });
            request("ListShortcuts", g_variant_new("(o@a{sv})", session_.c_str(), options({{"handle_token", token}})));
        });
        request("CreateSession", g_variant_new("(@a{sv})", options({{"handle_token", token},
            {"session_handle_token", "nemote_session_" + std::to_string(serial_)}})));
    }
    void edit_in_settings() {
        // Portal v1 has no ConfigureShortcuts. Open GNOME's application page,
        // whose shortcut editor updates the existing portal session. Never
        // unbind or overwrite an assignment merely to display an editor.
        configure_requested_ = false;
        error.clear();
        GError* launch_error = nullptr;
        auto* process = g_subprocess_new(G_SUBPROCESS_FLAGS_NONE, &launch_error,
            "gnome-control-center", "applications", "dev.nemote.Service", nullptr);
        if (!process) {
            error = std::string("Could not open GNOME Settings: ") + launch_error->message;
            g_error_free(launch_error);
            notify();
            return;
        }
        hint = "In GNOME Settings, open NeMote's shortcut settings on its Apps page.";
        notify();
        struct Launch { std::shared_ptr<PortalShortcuts> self; unsigned generation; };
        g_subprocess_wait_check_async(process, cancellable_, [](GObject* process, GAsyncResult* result, gpointer data) {
            std::unique_ptr<Launch> launch(static_cast<Launch*>(data));
            GError* error = nullptr;
            const bool success = g_subprocess_wait_check_finish(G_SUBPROCESS(process), result, &error);
            auto& self = *launch->self;
            if (!success && !self.stopped_ && launch->generation == self.generation_) {
                self.error = std::string("Could not open GNOME Settings: ") + error->message;
                self.notify(); // A launch failure must not unbind a working shortcut.
            }
            g_clear_error(&error);
        }, new Launch{shared_from_this(), generation_});
        g_object_unref(process);
    }
    void initialize() {
        if (stopped_ || busy_) return;
        busy_ = true;
        status = "connecting";
        error.clear();
        notify();
        auto discover = [this](GVariant*) {
            registered_ = true;
            call("org.freedesktop.DBus.Properties", "Get",
                g_variant_new("(ss)", interface, "version"), [this](GVariant* reply) {
                    GVariant* value;
                    g_variant_get(reply, "(v)", &value);
                    version_ = g_variant_get_uint32(value);
                    g_variant_unref(value);
                    if (version_ < 1) { fail("The desktop does not support the Global Shortcuts portal."); return; }
                    create_session();
                });
        };
        if (registered_) discover(nullptr);
        else call("org.freedesktop.host.portal.Registry", "Register",
            g_variant_new("(s@a{sv})", "dev.nemote.Service", options()), discover);
    }
public:
    bool setup_completed = false;
    std::string status = "unavailable", description, error, hint;
    PortalShortcuts(GDBusConnection* bus, std::function<void()> changed, std::function<void(bool)> activation)
        : bus_(G_DBUS_CONNECTION(g_object_ref(bus))), changed_(std::move(changed)), activation_(std::move(activation)) {}
    ~PortalShortcuts() { stop(); g_clear_object(&cancellable_); g_object_unref(bus_); }
    void start() {
        signals_ = g_dbus_connection_signal_subscribe(bus_, destination, interface, nullptr, path, nullptr,
            G_DBUS_SIGNAL_FLAGS_NONE, [](GDBusConnection*, const char*, const char*, const char*, const char* signal,
                                       GVariant* parameters, gpointer data) {
                auto& self = *static_cast<PortalShortcuts*>(data);
                auto* first = g_variant_get_child_value(parameters, 0);
                const bool ours = self.session_ == g_variant_get_string(first, nullptr);
                g_variant_unref(first);
                if (!ours) return;
                if (std::string(signal) == "ShortcutsChanged") {
                    self.hint.clear();
                    self.error.clear();
                    auto* list = g_variant_get_child_value(parameters, 1);
                    GVariantBuilder results;
                    g_variant_builder_init(&results, G_VARIANT_TYPE_VARDICT);
                    g_variant_builder_add(&results, "{sv}", "shortcuts", list);
                    auto* dict = g_variant_ref_sink(g_variant_builder_end(&results));
                    self.read_shortcuts(dict);
                    g_variant_unref(dict); g_variant_unref(list);
                    self.notify();
                } else if (self.bound_ && (std::string(signal) == "Activated" || std::string(signal) == "Deactivated")) {
                    auto* id = g_variant_get_child_value(parameters, 1);
                    if (std::string(g_variant_get_string(id, nullptr)) == "dictation")
                        self.activation_(std::string(signal) == "Activated");
                    g_variant_unref(id);
                }
            }, this, nullptr);
        responses_ = g_dbus_connection_signal_subscribe(bus_, destination, "org.freedesktop.portal.Request", "Response",
            nullptr, nullptr, G_DBUS_SIGNAL_FLAGS_NONE,
            [](GDBusConnection*, const char*, const char* object, const char*, const char*, GVariant* parameters, gpointer data) {
                auto& self = *static_cast<PortalShortcuts*>(data);
                if (self.request_path_ != object || !self.response_) return;
                guint code;
                GVariant* results;
                g_variant_get(parameters, "(u@a{sv})", &code, &results);
                auto callback = std::move(self.response_);
                self.clear_request();
                if (code == 0) callback(results);
                else self.fail(code == 1 ? "Shortcut setup was cancelled. Choose Configure to try again." : "The desktop could not configure the shortcut.");
                g_variant_unref(results);
            }, this, nullptr);
        closed_ = g_dbus_connection_signal_subscribe(bus_, destination, "org.freedesktop.portal.Session", "Closed",
            nullptr, nullptr, G_DBUS_SIGNAL_FLAGS_NONE,
            [](GDBusConnection*, const char*, const char* object, const char*, const char*, GVariant*, gpointer data) {
                auto& self = *static_cast<PortalShortcuts*>(data);
                if (self.session_ == object) self.fail("The desktop closed the shortcut session. Configure it again in Preferences.");
            }, this, nullptr);
        watch_ = g_bus_watch_name_on_connection(bus_, destination, G_BUS_NAME_WATCHER_FLAGS_AUTO_START,
            [](GDBusConnection*, const char*, const char*, gpointer data) {
                auto& self = *static_cast<PortalShortcuts*>(data);
                self.owner_present_ = true;
                self.initialize();
            }, [](GDBusConnection*, const char*, gpointer data) {
                auto& self = *static_cast<PortalShortcuts*>(data);
                self.owner_present_ = false;
                self.reset(); self.registered_ = false;
                self.status = "unavailable";
                self.error = "Global Shortcuts portal is unavailable. Install or restart xdg-desktop-portal and xdg-desktop-portal-gnome.";
                self.notify();
            }, this, nullptr);
    }
    void configure() {
        if (busy_) return;
        configure_requested_ = true;
        // The name watcher activates the portal. Wait for its owner callback
        // rather than racing Registry.Register against initial owner discovery.
        if (!owner_present_) return;
        if (session_.empty()) { initialize(); return; }
        if (!bound_) { bind(); return; }
        if (version_ == 1) { edit_in_settings(); return; }
        call(interface, "ConfigureShortcuts", g_variant_new("(os@a{sv})", session_.c_str(), "", options()),
            [this](GVariant*) { configure_requested_ = false; });
    }
    void stop() {
        if (stopped_) return;
        stopped_ = true;
        if (watch_) g_bus_unwatch_name(watch_);
        for (auto id : {signals_, responses_, closed_})
            if (id) g_dbus_connection_signal_unsubscribe(bus_, id);
        reset();
        changed_ = nullptr;
        activation_ = nullptr;
    }
};
