import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import React from 'react';

import { dateTime, dateTimeFormat } from '@grafana/data';
import { Components } from '@grafana/e2e-selectors';

import { render } from '../../../test/test-utils';

import { BannerForm } from './BannerForm';
import { NotificationBannerSpec } from './api/types';

import '@testing-library/jest-dom';

jest.mock('@grafana/runtime', () => ({
  ...jest.requireActual('@grafana/runtime'),
  getAppEvents: jest.fn(() => ({
    publish: jest.fn(),
  })),
  config: {
    namespace: 'default',
    theme2: { breakpoints: { values: {} } },
    featureToggles: {
      notificationBanner: true,
    },
  },
}));

jest.mock('@grafana/ui', () => ({
  ...jest.requireActual('@grafana/ui'),
  CodeEditor: function CodeEditor({ value, onBlur }: { value: string; onBlur: (newValue: string) => void }) {
    return <input data-testid="mockeditor" value={value} onChange={(e) => onBlur(e.currentTarget.value)} />;
  },
}));

function setup(jsx: JSX.Element) {
  return {
    user: userEvent.setup(),
    ...render(jsx),
  };
}

const mockSubmitData = jest.fn();
const mockRequest = { isSuccess: false };

jest.mock('./hooks', () => ({
  useCreateOrUpdateBanner: jest.fn(() => [mockSubmitData, mockRequest]),
}));

describe('BannerForm', () => {
  it('renders with default values when no banner and name are provided', async () => {
    setup(<BannerForm />);
    const now = dateTimeFormat(dateTime(), { format: 'YYYY-MM-DD HH:mm' });
    const [startTime, endTime] = await screen.findAllByTestId(Components.DateTimePicker.input);
    expect(startTime).toHaveValue(now);
    expect(endTime).toHaveValue('');
    expect(screen.getByRole('checkbox', { name: /Enabled/ })).not.toBeChecked();
    expect(screen.getByTestId(/mockeditor/)).toHaveTextContent('');
    expect(screen.getByRole('radio', { name: /Everyone/ })).toBeChecked();
    expect(screen.getByRole('button', { name: /Save/ })).toBeInTheDocument();
  });

  it('correctly displays the provided banner values', async () => {
    const banner: NotificationBannerSpec = {
      enabled: true,
      message: 'Test message',
      variant: 'warning',
      visibility: 'authenticated',
      startTime: dateTime('2024-05-23 06:20').toISOString(),
      endTime: dateTime('2024-05-25 06:20').toISOString(),
    };
    setup(<BannerForm banner={banner} name={'test'} />);
    const [startTime, endTime] = await screen.findAllByTestId(Components.DateTimePicker.input);

    expect(screen.getByRole('checkbox', { name: /Enabled/ })).toBeChecked();
    expect(screen.getByTestId(/mockeditor/)).toHaveValue('Test message');
    expect(startTime).toHaveValue('2024-05-23 06:20');
    expect(endTime).toHaveValue('2024-05-25 06:20');
    expect(screen.getByRole('radio', { name: /Authenticated/ })).toBeChecked();
    expect(screen.getByRole('button', { name: /Save/ })).toBeInTheDocument();
  });

  it('should validate the end date', async () => {
    const { user } = setup(<BannerForm />);
    const [startTime, endTime] = screen.getAllByTestId(Components.DateTimePicker.input);

    await user.clear(startTime);
    await user.type(startTime, '2024-05-24T06:35:00.000Z');
    await user.type(endTime, '2024-05-23T06:20:00.000Z');
    await user.click(screen.getByRole('button', { name: /Save/ }));

    await waitFor(() => {
      expect(mockSubmitData).not.toHaveBeenCalled();
    });
    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText('End time must be after start time')).toBeInTheDocument();
  });

  it('calls submitData with updated field values when the form is submitted', async () => {
    const { user } = setup(<BannerForm />);
    const [startTime, endTime] = await screen.findAllByTestId(Components.DateTimePicker.input);
    await user.click(screen.getByRole('checkbox', { name: /Enabled/ }));
    await user.clear(startTime);
    await user.type(startTime, '2024-05-24T06:35:00.000Z');
    await user.type(endTime, '2024-07-24T06:35:00.000Z');
    await user.click(screen.getByRole('radio', { name: /Authenticated/ }));
    await user.click(screen.getByRole('button', { name: /Save/ }));

    await waitFor(() => {
      expect(mockSubmitData).toHaveBeenCalledWith({
        enabled: true,
        message: '',
        startTime: '2024-05-24T06:35:00.000Z',
        endTime: '2024-07-24T06:35:00.000Z',
        visibility: 'authenticated',
        variant: 'info',
      });
    });
  });
});
