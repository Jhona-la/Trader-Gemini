import { DateTime } from '@grafana/data';

export type NotificationBannerVisibility = 'authenticated' | 'everyone';
export type NotificationBannerFormData = {
  enabled: boolean;
  startTime?: DateTime | null;
  endTime?: DateTime | null;
  message: string;
  variant: 'error' | 'info' | 'warning';
  visibility: NotificationBannerVisibility;
};
