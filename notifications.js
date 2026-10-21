// Notification System JavaScript
// Polls for new notifications and updates the navbar bell icon

let lastUnreadCount = 0;

async function fetchNotifications() {
    try {
        const res = await fetch('/api/notifications?limit=5');
        if (!res.ok) return;
        const data = await res.json();
        updateNotificationUI(data.notifications, data.unread_count);
    } catch (e) {
        console.error('Notification fetch error:', e);
    }
}

function updateNotificationUI(notifications, unreadCount) {
    const badge = document.getElementById('notificationBadge');
    const list = document.getElementById('notificationList');

    // Update badge
    if (badge) {
        if (unreadCount > 0) {
            badge.textContent = unreadCount > 99 ? '99+' : unreadCount;
            badge.classList.remove('d-none');
        } else {
            badge.classList.add('d-none');
        }
    }

    // Update list
    if (list) {
        if (notifications.length === 0) {
            list.innerHTML = '<li class="dropdown-item text-center text-muted py-3">No notifications</li>';
        } else {
            list.innerHTML = notifications.map(n => `
                <li class="dropdown-item d-flex justify-content-between align-items-start ${n.is_read ? '' : 'bg-light'}">
                    <div class="me-2" style="cursor:pointer" onclick="handleNotificationClick(${n.id}, '${n.link || ''}')">
                        <div class="fw-bold small">
                            ${n.type === 'critical' ? '<i class="fas fa-exclamation-circle text-danger me-1"></i>' : 
                              n.type === 'warning' ? '<i class="fas fa-exclamation-triangle text-warning me-1"></i>' : 
                              '<i class="fas fa-info-circle text-info me-1"></i>'}
                            ${n.message}
                        </div>
                        <div class="text-muted" style="font-size: 0.75rem;">
                            ${n.category} &bull; ${new Date(n.created_at).toLocaleString()}
                        </div>
                    ${!n.is_read ? '<span class="badge bg-primary rounded-pill" style="font-size:0.6rem;">New</span>' : ''}
                </li>
            `).join('');
        }
    }

    // Browser notification for critical alerts
    if (unreadCount > lastUnreadCount) {
        const critical = notifications.find(n => n.type === 'critical' && !n.is_read);
        if (critical && 'Notification' in window && Notification.permission === 'granted') {
            new Notification('IVF Journey Tracker Alert', {
                body: critical.message,
                icon: '/static/favicon.ico'
            });
        }
    }
    lastUnreadCount = unreadCount;
}

async function handleNotificationClick(id, link) {
    try {
        await fetch(`/api/notifications/${id}/read`, { method: 'POST' });
        if (link) {
            window.location.href = link;
        } else {
            fetchNotifications();
        }
    } catch (e) {
        console.error(e);
    }
}

async function markAllNotificationsRead() {
    try {
        const res = await fetch('/api/notifications/read_all', { method: 'POST' });
        if (res.ok) {
            fetchNotifications();
        }
    } catch (e) {
        console.error(e);
    }
}

// Request browser notification permission on load
if ('Notification' in window && Notification.permission === 'default') {
    Notification.requestPermission();
}

// Initial fetch and polling
document.addEventListener('DOMContentLoaded', () => {
    fetchNotifications();
    setInterval(fetchNotifications, 30000); // Poll every 30 seconds
});
