import { css } from '@emotion/css';
import React from 'react';
import { Controller, useForm } from 'react-hook-form';

import { GrafanaTheme2 } from '@grafana/data';
import { Button, Input, Field, Stack, useStyles2 } from '@grafana/ui';
import { CloseButton } from 'app/core/components/CloseButton/CloseButton';
import { TeamPicker } from 'app/core/components/Select/TeamPicker';
import { Trans } from 'app/core/internationalization';

import { validateLBACRule } from './utils';

export interface LBACFormData {
  team: number;
  rule: string;
}

interface AddTeamLBACFormProps {
  onSubmit: (data: LBACFormData) => void;
  onClose: () => void;
}

export const CreateTeamLBACForm = ({ onSubmit, onClose }: AddTeamLBACFormProps) => {
  const styles = useStyles2(getStyles);
  const {
    handleSubmit,
    register,
    control,
    formState: { errors },
  } = useForm<LBACFormData>({});
  return (
    <div className="cta-form" aria-label="Permissions slider">
      <CloseButton onClick={onClose} />
      <h5>Create LBAC Rule</h5>

      <form name="addTeamLBAC" onSubmit={handleSubmit(onSubmit)}>
        <Stack gap={1} direction="row">
          <Field label="Team">
            <Controller
              render={({ field: { onChange, ref, ...field } }) => (
                <TeamPicker {...field} onSelected={(t) => onChange(t.value?.id)} />
              )}
              control={control}
              name={`team`}
              rules={{ required: { value: true, message: 'Required' } }}
            />
          </Field>
          <Field label="Rule" invalid={!!errors.rule} error={errors?.rule?.message}>
            <Input
              className={styles.input}
              type="text"
              placeholder='{label="foo", label2="bar"}'
              {...register('rule', {
                required: 'Rule is required',
                validate: (val) => validateLBACRule(val) || 'Invalid LBAC rule syntax',
              })}
              width={30}
              invalid={!!errors.rule}
            />
          </Field>
          <Field label="">
            <Button type="submit">
              <Trans i18nKey="access-control.add-permissions.save">Save</Trans>
            </Button>
          </Field>
        </Stack>
      </form>
    </div>
  );
};

const getStyles = (theme: GrafanaTheme2) => {
  return {
    input: css({
      '& input': {
        fontFamily: theme.typography.fontFamilyMonospace,
      },
    }),
  };
};
