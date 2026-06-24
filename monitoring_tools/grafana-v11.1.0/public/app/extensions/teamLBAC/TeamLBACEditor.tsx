import { css } from '@emotion/css';
import { isEmpty } from 'lodash';
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { ConnectedProps, connect } from 'react-redux';

import { DataSourceJsonData, DataSourceSettings, GrafanaTheme2 } from '@grafana/data';
import { ConfigSection } from '@grafana/experimental';
import { getBackendSrv } from '@grafana/runtime';
import { Alert, Button, Icon, Stack, useStyles2 } from '@grafana/ui';
import { ResourcePermission } from 'app/core/components/AccessControl/types';
import { contextSrv } from 'app/core/core';
import { DataSourceReadOnlyMessage } from 'app/features/datasources/components/DataSourceReadOnlyMessage';
import { Team } from 'app/types';

import { AccessControlAction as EnterpriseActions, EnterpriseStoreState, TeamRule } from '../types';

import { CreateTeamLBACForm, LBACFormData } from './AddTeamLBACForm';
import { TeamRulesRow } from './TeamRulesRow';
import { getTeamLBAC, updateTeamLBACRules } from './state/actions';
import { formatLBACRule } from './utils';

export interface OwnProps {
  dataSourceConfig: DataSourceSettings<DataSourceJsonData, {}>;
  readOnly?: boolean;
  onTeamLBACUpdate: () => Promise<DataSourceSettings> | void;
  getWarnings?: (items: TeamRule[]) => TeamRule[];
  items?: ResourcePermission[];
}

function mapStateToProps(state: EnterpriseStoreState) {
  return {
    teamLBACConfig: state.teamLBAC?.teamLBACConfig,
  };
}

const mapDispatchToProps = {
  getTeamLBAC,
  updateTeamLBACRules,
};

export const connector = connect(mapStateToProps, mapDispatchToProps);
type Props = OwnProps & ConnectedProps<typeof connector>;

export const TeamLBACEditorUnconnected = ({
  teamLBACConfig,
  dataSourceConfig,
  readOnly,
  onTeamLBACUpdate,
  getTeamLBAC,
  updateTeamLBACRules,
  items,
}: Props) => {
  const [teams, setTeams] = useState<Array<Pick<Team, 'name' | 'avatarUrl' | 'id'>>>([]);
  const [showLBACForm, setShowLBACForm] = useState(false);
  const [rolesWithoutLBAC, setRolesWithoutLBAC] = useState<Set<string>>(new Set<string>());
  const [serviceAccountsWithoutLBAC, setServiceAccountsWithoutLBAC] = useState<Set<number>>(new Set<number>());
  const [teamsWithoutLBAC, setTeamsWithoutLBAC] = useState<Set<string>>(new Set<string>());
  const [lbacTeamsWithoutTeamPermissions, setLBACTeamsWithoutTeamPermission] = useState<Set<string>>(new Set<string>());
  const lbacRules = useMemo(() => teamLBACConfig?.rules || [], [teamLBACConfig?.rules]);
  const styles = useStyles2(getStyles);

  useEffect(() => {
    getTeamLBAC(dataSourceConfig.uid);
  }, [dataSourceConfig.uid, getTeamLBAC]);

  useEffect(() => {
    const fetchTeams = async () => {
      const teamIds = lbacRules.map((r) => r.teamId);
      if (!teamIds?.length) {
        return;
      }

      // TODO: to only get the teams which have query permissions to the datasource
      const result = await getBackendSrv().get('/api/teams/search', { teamId: teamIds });
      const teamsArray: Team[] = result?.teams;
      const teams = teamsArray.map((team) => ({
        id: team.id,
        value: team.id,
        teamId: team.id,
        name: team.name,
        avatarUrl: team.avatarUrl,
      }));
      setTeams(teams);
    };
    fetchTeams();
  }, [lbacRules]);

  useEffect(() => {
    if (!items?.length) {
      return;
    }
    const lbacTeams = new Set(lbacRules.map((r) => r.teamId));
    const resourceTeams = new Set(items.map((r) => r.teamId));
    const resourceServiceAccounts: Set<number> = new Set(items.filter((r) => r.isServiceAccount).map((r) => r.id));
    const resourceRoles: Set<string> = new Set(
      items
        .filter((r) => typeof r.builtInRole === 'string' && r.builtInRole.trim() !== '' && r.permission !== 'Admin')
        .map((r) => r.builtInRole as string) // Assert that r.builtInRole is a string
    );

    const teamsWithoutLbac: Set<string> = new Set(
      [...resourceTeams].filter((x) => x !== undefined && !lbacTeams.has(x.toString())).map((x) => x?.toString() || '')
    );
    const lbacTeamsWithoutTeamPermissions = new Set([...lbacTeams].filter((x) => x && !resourceTeams.has(Number(x))));

    if (lbacTeamsWithoutTeamPermissions) {
      setLBACTeamsWithoutTeamPermission(lbacTeamsWithoutTeamPermissions);
    }
    if (teamsWithoutLbac) {
      setTeamsWithoutLBAC(teamsWithoutLbac);
    }
    if (resourceServiceAccounts) {
      setServiceAccountsWithoutLBAC(resourceServiceAccounts);
    }
    if (resourceRoles) {
      setRolesWithoutLBAC(resourceRoles);
    }
  }, [items, lbacRules]);

  const onTeamLBACUpdateInternal = useCallback(
    async (rules: TeamRule[]) => {
      await updateTeamLBACRules(dataSourceConfig.uid, { rules });
      await onTeamLBACUpdate();
    },
    [dataSourceConfig.uid, onTeamLBACUpdate, updateTeamLBACRules]
  );

  const onSubmitLBAC = async ({ team, rule }: LBACFormData) => {
    let updatedRules: TeamRule[] = [];
    const existingTeamRules = lbacRules.find((teamRules) => teamRules.teamId === team.toString());
    if (existingTeamRules) {
      updatedRules = lbacRules.map((teamRules) => {
        if (teamRules.teamId === team.toString()) {
          return { ...teamRules, rules: [...teamRules.rules, formatLBACRule(rule)] };
        }
        return { ...teamRules };
      });
    } else {
      updatedRules = lbacRules.concat({ teamId: team.toString(), rules: [formatLBACRule(rule)] });
    }
    await onTeamLBACUpdateInternal(updatedRules);
    setShowLBACForm(false);
  };

  const onRulesUpdate = async (teamId: string, teamRules: string[]) => {
    const updatedRules: TeamRule[] = [];
    lbacRules.forEach((r) => {
      if (r.teamId === teamId) {
        if (!teamRules?.length) {
          return;
        }
        updatedRules.push({ ...r, rules: teamRules });
      } else {
        updatedRules.push({ ...r });
      }
    });
    await onTeamLBACUpdateInternal(updatedRules);
  };

  const canEdit = contextSrv.hasPermission(EnterpriseActions.DataSourcesPermissionsWrite) && !readOnly;
  const getDescription = (item: ResourcePermission) => {
    if (item.userId) {
      return item.userLogin;
    } else if (item.teamId) {
      return item.team;
    } else if (item.builtInRole) {
      return item.builtInRole;
    }
    return;
  };
  return (
    <div>
      {readOnly && <DataSourceReadOnlyMessage />}
      <Stack direction="column" gap={2}>
        <ConfigSection
          title="Data Access"
          description="Here you can configure access to specific data within the datasource. "
        >
          <>
            {(lbacTeamsWithoutTeamPermissions.size > 0 ||
              teamsWithoutLBAC.size > 0 ||
              serviceAccountsWithoutLBAC.size > 0 ||
              rolesWithoutLBAC.size > 0) &&
              lbacRules.length > 0 && (
                <div>
                  <Alert title={`Permission warning`} severity="warning">
                    Recommended setup is to allow team query permissions only.
                    <ul className={styles.warning}>
                      {lbacTeamsWithoutTeamPermissions.size > 0 && (
                        <div>Teams cannot query the data source. Add query permissions for the following teams:</div>
                      )}
                      {[...lbacTeamsWithoutTeamPermissions].map((team) => {
                        const teamName = teams?.find((t) => t.id?.toString() === team)?.name;
                        return (
                          <li className={styles.warningli} key={team}>
                            {`Team ${teamName}`}
                          </li>
                        );
                      })}
                      {(teamsWithoutLBAC.size > 0 || serviceAccountsWithoutLBAC.size > 0) && (
                        <div>Teams can query all logs. Add Team LBAC rules for the following teams:</div>
                      )}
                      {[...teamsWithoutLBAC].map((team) => {
                        const i = items?.find((t) => t.teamId?.toString() === team);
                        if (!i) {
                          return;
                        }
                        return <li className={styles.warningli} key={team}>{`Team ${getDescription(i)}`}</li>;
                      })}
                      {[...serviceAccountsWithoutLBAC].map((serviceaccount) => {
                        const sAID = items?.find((t) => t.id === serviceaccount);
                        if (!sAID) {
                          return;
                        }
                        return (
                          <li
                            className={styles.warningli}
                            key={serviceaccount}
                          >{`Service account ${getDescription(sAID)}`}</li>
                        );
                      })}
                      {rolesWithoutLBAC.size > 0 && (
                        <div>
                          Role can access all logs with `query`` permission. Remove permission for the following:
                        </div>
                      )}
                      {[...rolesWithoutLBAC].map((role) => {
                        return <li className={styles.warningli} key={role}>{`Role ${role}`}</li>;
                      })}
                    </ul>
                  </Alert>
                </div>
              )}
          </>
          <Stack direction="column" alignItems={'start'} gap={2}>
            <h4>Team Label-based access control</h4>
            <ul className={styles.warning}>
              <Icon name="info" /> Note: Feature preview notes for Team LBAC rules will not be applied if the linked
              Cloud Access Policy has label selectors.
            </ul>
            Here you can configure label-based access control (LBAC) rules for teams, <br></br>
            allowing you to restrict which logs a team can query based on logql.
            <ul className={css({ marginLeft: 16 })}>
              <li>
                Ex: A rule <code>cluster=&quot;us-west-0&quot;, namespace=~&quot;dev|prod&quot;</code>
              </li>
            </ul>
            {canEdit && <Button onClick={() => setShowLBACForm(true)}>Create LBAC rule</Button>}
            {showLBACForm && <CreateTeamLBACForm onSubmit={onSubmitLBAC} onClose={() => setShowLBACForm(false)} />}
          </Stack>
        </ConfigSection>
        {!isEmpty(lbacRules) && (
          <table role="grid" className="filter-table gf-form-group" aria-labelledby="team_lbac_rules">
            <thead>
              <tr>
                <th style={{ width: '30%' }}>Team</th>
                <th style={{ width: '1%' }} />
                <th>Label selector</th>
                <th style={{ width: '1%' }} />
              </tr>
            </thead>
            <tbody>
              {lbacRules.map((teamRules, idx) => {
                const rules = teamRules.rules;
                const teamId = teamRules.teamId;
                let warning = '';
                if (lbacTeamsWithoutTeamPermissions.has(teamRules.teamId)) {
                  warning = 'Warning: This team may not have permission to query the datasource.';
                }
                return (
                  <TeamRulesRow
                    key={idx}
                    teamRules={rules}
                    disabled={!canEdit}
                    team={
                      teams.find((t) => t.id?.toString() === teamId) || {
                        id: Number(teamId),
                        name: '',
                        avatarUrl: '',
                      }
                    }
                    onChange={(teamRules) => onRulesUpdate(teamId, teamRules)}
                    warning={warning}
                  />
                );
              })}
            </tbody>
          </table>
        )}
      </Stack>
    </div>
  );
};

const getStyles = (theme: GrafanaTheme2) => {
  return {
    wrapper: css({
      display: 'flex',
      flexDirection: 'column',
      overflowX: 'auto',
      overflowY: 'hidden',
      width: '100%',
    }),
    sectionHeader: css({
      marginBottom: theme.spacing(2),
    }),
    warning: css({
      fontSize: theme.typography.bodySmall.fontSize,
    }),
    warningli: css({
      listStyle: 'inside',
    }),
  };
};

export const TeamLBACEditor = connector(TeamLBACEditorUnconnected);
