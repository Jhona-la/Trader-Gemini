import { Resource, ResourceForCreate, ResourceList } from 'app/features/apiserver/types';

import { baseAPI as api } from './baseAPI';
import { RequestArg, NotificationBannerSpec, UpdateRequestArg } from './types';

const injectedRtkApi = api.injectEndpoints({
  endpoints: (build) => ({
    listNotificationBanner: build.query<ResourceList<NotificationBannerSpec>, void>({
      query: () => ({
        url: `/notificationbanners`,
      }),
      providesTags: ['NotificationBannerList'],
    }),
    createNotificationBanner: build.mutation<void, ResourceForCreate<NotificationBannerSpec>>({
      query: (resource) => ({
        url: `/notificationbanners`,
        method: 'POST',
        body: resource,
      }),
      invalidatesTags: ['NotificationBannerList'],
    }),
    getNotificationBanner: build.query<Resource<NotificationBannerSpec>, RequestArg>({
      query: (queryArg) => ({ url: `/notificationbanners/${queryArg.name}` }),
    }),
    deleteNotificationBanner: build.mutation<unknown, RequestArg>({
      query: (queryArg) => ({
        url: `/notificationbanners/${queryArg.name}`,
        method: 'DELETE',
      }),
    }),
    updateNotificationBanner: build.mutation<
      Resource<NotificationBannerSpec>,
      UpdateRequestArg<NotificationBannerSpec>
    >({
      query: (queryArg) => ({
        url: `/notificationbanners/${queryArg.name}`,
        method: 'PATCH',
        headers: { 'Content-Type': 'application/merge-patch+json' },
        body: JSON.stringify(queryArg.body),
      }),
      invalidatesTags: ['NotificationBannerList'],
    }),
  }),
  overrideExisting: false,
});
export { injectedRtkApi as generatedAPI };

export const {
  useListNotificationBannerQuery,
  useCreateNotificationBannerMutation,
  useGetNotificationBannerQuery,
  useDeleteNotificationBannerMutation,
  useUpdateNotificationBannerMutation,
} = injectedRtkApi;
