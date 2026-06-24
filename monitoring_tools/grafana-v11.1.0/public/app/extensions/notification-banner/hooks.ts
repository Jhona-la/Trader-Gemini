import { useCallback } from 'react';

import {
  useCreateNotificationBannerMutation,
  useListNotificationBannerQuery,
  useUpdateNotificationBannerMutation,
} from './api';
import { NotificationBannerSpec } from './api/types';

export function useCreateOrUpdateBanner(name?: string) {
  const [create, createRequest] = useCreateNotificationBannerMutation();
  const [update, updateRequest] = useUpdateNotificationBannerMutation();

  const updateOrCreate = useCallback(
    (data: NotificationBannerSpec) => {
      if (name) {
        return update({ name, body: { metadata: { name }, spec: data } });
      }
      return create({ metadata: { generateName: 'banner' }, spec: data });
    },
    [create, name, update]
  );
  return [updateOrCreate, name ? updateRequest : createRequest] as const;
}

interface Banner {
  spec: NotificationBannerSpec;
  metadata: { name: string };
}

export function useBanner(): [Banner | undefined, boolean] {
  const query = useListNotificationBannerQuery();
  return [query.data?.items?.at(-1), query.isLoading];
}
