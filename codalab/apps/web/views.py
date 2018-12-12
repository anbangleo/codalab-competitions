# coding=utf-8
import csv
import urllib
from datetime import datetime, timedelta
import json
import math
import os
import StringIO
import sys
import traceback
import yaml
import zipfile
import copy

from decimal import Decimal

from django.views.generic.base import ContextMixin
from yaml.representer import SafeRepresenter

from django.db import connection
from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.contrib.sites.models import Site
from django.core.cache import cache
from django.core.exceptions import ImproperlyConfigured
from django.core.exceptions import ObjectDoesNotExist
from django.core.exceptions import PermissionDenied
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.core.servers.basehttp import FileWrapper
from django.core.urlresolvers import reverse, reverse_lazy
from django.db.models import Q, Max, Min, Count
from django.http import Http404, HttpResponseForbidden
from django.http import HttpResponse, HttpResponseRedirect
from django.http import StreamingHttpResponse
from django.shortcuts import render_to_response, render, redirect
from django.template import RequestContext, loader
from django.utils.decorators import method_decorator
from django.utils.html import strip_tags
from django.views.generic import FormView
from django.views.generic import View, TemplateView, DetailView, ListView, UpdateView, CreateView, DeleteView
from django.utils.html import strip_tags
from django.utils import timezone

#hhm add 2018.5.4
from apps.authenz.models import ClUser

from mimetypes import MimeTypes

from apps.teams.forms import OrganizerTeamsCSVForm
from apps.jobs.models import Job
from apps.web import forms
from apps.web import models
from apps.web import tasks
from apps.coopetitions.models import Like, Dislike
from apps.forums.models import Forum
from apps.common.competition_utils import get_most_popular_competitions, get_featured_competitions
from apps.web.exceptions import ScoringException
from apps.web.forms import CompetitionS3UploadForm, SubmissionS3UploadForm,DocumentForm
from apps.web.models import SubmissionScore, SubmissionScoreDef, get_current_phase, \
    get_first_previous_active_and_next_phases, Document, Competition


from apps.authenz.models import ClUser
from tasks import evaluate_submission, re_run_all_submissions_in_phase, create_competition, _make_url_sassy, \
    make_modified_bundle
from apps.teams.models import Team, TeamMembership, get_user_team, get_competition_teams, get_competition_pending_teams, get_competition_deleted_teams, get_last_team_submissions, get_user_requests, get_team_pending_membership

from extra_views import UpdateWithInlinesView, InlineFormSet, NamedFormsetsMixin

from .utils import check_bad_scores
from django.shortcuts import render

from binaryclass import BinaryClassTest

try:
    import azure
    import azure.storage
except ImportError:
    raise ImproperlyConfigured(
        "Could not load Azure bindings. "
        "See https://github.com/WindowsAzure/azure-sdk-for-python")

User = get_user_model()


iftrain = 0
iftest = 0
ifunlabel = 0
ifdataset = 0
numofdataset = 0
numofunlabel = 0
ifzip = 0

############################################################
# General: template views



class MyAdminView(TemplateView):
    """Admin page for monitoring services"""
    template_name = "web/admin_monitoring_links.html"

    def get(self, *args, **kwargs):
        redirect_url = "index.html"
        user = self.request.user
        if user.is_staff and user.is_active:
            return super(MyAdminView, self).get(*args, **kwargs)
        else:
            return HttpResponseRedirect(redirect_url)

    def get_context_data(self, **kwargs):
        """Used to grab context in Class Based Views"""
        context = super(MyAdminView, self).get_context_data(**kwargs)
        context["domain"] = Site.objects.get_current().domain
        context["rabbit_port"] = settings.RABBITMQ_MANAGEMENT_PORT
        context["flower_port"] = settings.FLOWER_PORT
        return context


class HomePageView(TemplateView):
    """Template View for homepage."""
    template_name = "web/index.html"

    def get(self, *args, **kwargs):
        if settings.SINGLE_COMPETITION_VIEW_PK:
            # First quickly check that competition is available to view
            try:
                competition = models.Competition.objects.get(pk=settings.SINGLE_COMPETITION_VIEW_PK)
                if not competition.published:
                    return HttpResponse(
                        "Warning, SINGLE_COMPETITION_VIEW_PK setting is set but the competition is not published so "
                        "regular users won't be able to access it!<br>"
                        "If you have access, go <a href='{}'>here</a> to edit the competition.".format(
                            reverse("competitions:edit", kwargs={"pk": competition.pk})
                        ))
            except ObjectDoesNotExist:
                raise Http404()

            kwargs.update(pk=settings.SINGLE_COMPETITION_VIEW_PK)
            return CompetitionDetailView.as_view()(*args, **kwargs)
        return super(HomePageView, self).get(*args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super(HomePageView, self).get_context_data(**kwargs)

        c_key = 'popular_competitions'
        popular_competitions = cache.get(c_key)
        if not popular_competitions:
            popular_competitions = get_most_popular_competitions()
            cache.set(c_key, popular_competitions, 60 * 60 * 1)

        context['latest_competitions'] = popular_competitions
        context['featured_competitions'] = get_featured_competitions()

        return context


class LoginRequiredMixin(object):
    @method_decorator(login_required)
    def dispatch(self, *args, **kwargs):
        return super(LoginRequiredMixin, self).dispatch(*args, **kwargs)


class UserSettingsView(LoginRequiredMixin, UpdateView):
    """View to display User settings."""
    template_name = "web/my/settings.html"
    form_class = forms.UserSettingsForm
    model = User
    success_url = "/my/settings/"

    def get_object(self, queryset=None):
        return self.request.user

class DeveloperLab(LoginRequiredMixin, UpdateView):
    """View to display ALLab """
    template_name = "web/my/developerlab.html"
    form_class = forms.UserSettingsForm
    model = User
    success_url = "/my/settings/"

    def get_object(self, queryset=None):
        return self.request.user


############################################################
# Competitions: template views

def my_documents(request):
    return render(request, 'web/competitions/document/docs.html')

def competition_index(request):
    """
    View that list all competitions.

    .. note::

        Two different queries:
            - is_active
            - is_finished

    """
    query = request.GET.get('q')
    is_active = request.GET.get('is_active', False)
    is_finished = request.GET.get('is_finished', False)
    medical_image_viewer = request.GET.get('medical_image_viewer', False)

    competitions = models.Competition.objects.filter(published=True)

    if query:
        competitions = competitions.filter(Q(title__iregex=".*%s" % query) | Q(description__iregex=".*%s" % query))
    if medical_image_viewer:
        competitions = competitions.filter(enable_medical_image_viewer=True)
    if is_active:
        competitions = [c for c in competitions if c.is_active]
    if is_finished:
        competitions = [c for c in competitions if not c.is_active]

    competitions = reversed(sorted(competitions, key=lambda c: c.get_start_date))

    return render(request, "web/competitions/index.html", {
        'competitions': competitions,
    })



@login_required
def my_index(request):
    """
    View that lists competitions created by an user.

    .. note::

        - User needs to be authenticated.
    """
    template = loader.get_template("web/my/index.html")
    try:
        denied = models.ParticipantStatus.objects.get(codename=models.ParticipantStatus.DENIED)
    except:
        denied = -1

    my_competitions = models.Competition.objects.filter(Q(creator=request.user) | Q(admins__in=[request.user])).order_by('-pk').select_related('creator').distinct()
    published_competitions = models.Competition.objects.filter(published=True).select_related('creator', 'participants')
    published_competitions = reversed(sorted(published_competitions, key=lambda c: c.get_start_date))
    context_dict = {
        'my_competitions': my_competitions,
        'competitions_im_in': list(request.user.participation.all().exclude(status=denied).select_related('creator')),
        'published_competitions': published_competitions,
    }
    return HttpResponse(template.render(RequestContext(request, context_dict)))


def sort_data_table(request, context, list):
    context['order'] = order = request.GET.get('order') if 'order' in request.GET else 'id'
    context['direction'] = direction = request.GET.get('direction') if 'direction' in request.GET else 'asc'
    reverse = direction == 'desc'

    def sortkey(x):
        return x[order] if order in x and x[order] is not None else ''
    list.sort(key=sortkey, reverse=reverse)


#
# Competition Views
#

class PhasesInline(InlineFormSet):
    model = models.CompetitionPhase
    form_class = forms.CompetitionPhaseForm
    extra = 0


class PagesInline(InlineFormSet):
    model = models.Page
    form_class = forms.PageForm
    extra = 0


class LeaderboardInline(InlineFormSet):
    model = models.SubmissionScoreDef
    form_class = forms.LeaderboardForm
    extra = 0


class CompetitionUpload(LoginRequiredMixin, CreateView):
    model = models.CompetitionDefBundle
    template_name = 'web/competitions/upload_competition.html'


class CompetitionS3Upload(LoginRequiredMixin, FormView):
    form_class = CompetitionS3UploadForm
    template_name = 'web/competitions/upload_s3_competition.html'

    def form_valid(self, form):
        competition_def_bundle = form.save(commit=False)
        competition_def_bundle.owner = self.request.user
        competition_def_bundle.save()
        job = Job.objects.create_job('create_competition', {'comp_def_id': competition_def_bundle.pk})
        create_competition.apply_async((job.pk, competition_def_bundle.pk,))
        return HttpResponse(json.dumps({'token': job.pk}), status=201, content_type="application/json")


class CompetitionEdit(LoginRequiredMixin, NamedFormsetsMixin, UpdateWithInlinesView):
    """ View to edit a competition"""
    model = models.Competition
    form_class = forms.CompetitionForm
    inlines = [PagesInline, PhasesInline, LeaderboardInline]
    inlines_names = ['Pages', 'Phases', 'Leaderboards']
    template_name = 'web/competitions/edit.html'

    def get_form_kwargs(self):
        kwargs = super(CompetitionEdit, self).get_form_kwargs()
        kwargs['user'] = self.request.user
        return kwargs

    def forms_valid(self, form, inlines):
        form.instance.modified_by = self.request.user

        # save up here, before checks for new phase data
        save_result = super(CompetitionEdit, self).forms_valid(form, inlines)

        # inlines[0] = pages
        # inlines[1] = phases
        for phase_form in inlines[1]:
            if phase_form.cleaned_data.get("input_data_organizer_dataset"):
                phase_form.instance.input_data = phase_form.cleaned_data["input_data_organizer_dataset"].data_file.file.name

            if phase_form.cleaned_data.get("reference_data_organizer_dataset"):
                phase_form.instance.reference_data = phase_form.cleaned_data["reference_data_organizer_dataset"].data_file.file.name

            if phase_form.cleaned_data.get("scoring_program_organizer_dataset"):
                phase_form.instance.scoring_program = phase_form.cleaned_data["scoring_program_organizer_dataset"].data_file.file.name

            if phase_form.cleaned_data.get("ingestion_program_organizer_dataset"):
                phase_form.instance.ingestion_program = phase_form.cleaned_data["ingestion_program_organizer_dataset"].data_file.file.name

            phase_form.instance.save()

        # Look for admins that are not participants yet
        approved_status = models.ParticipantStatus.objects.get(codename=models.ParticipantStatus.APPROVED)

        for admin in form.instance.admins.all():
            try:
                participant = models.CompetitionParticipant.objects.get(user=admin, competition=form.instance)
                participant.status = approved_status
                participant.save()
            except ObjectDoesNotExist:
                models.CompetitionParticipant.objects.create(user=admin, competition=form.instance, status=approved_status)

        return save_result

    def get_context_data(self, **kwargs):
        context = super(CompetitionEdit, self).get_context_data(**kwargs)
        return context

    def construct_inlines(self):
        '''I need to overwrite this method in order to change
        the queryset for the "keywords" field'''
        inline_formsets = super(CompetitionEdit, self).construct_inlines()

        # NOTE:
        #   inline_formsets[0] == web pages
        #   inline_formsets[1] == phases

        # get existing datasets and add them, so admins can see them!
        public_data_ids = models.CompetitionPhase.objects.filter(competition=self.object).values_list('public_data_organizer_dataset')
        starting_kit_ids = models.CompetitionPhase.objects.filter(competition=self.object).values_list('starting_kit_organizer_dataset')
        input_data_ids = models.CompetitionPhase.objects.filter(competition=self.object).values_list('input_data_organizer_dataset')
        reference_data_ids = models.CompetitionPhase.objects.filter(competition=self.object).values_list('reference_data_organizer_dataset')
        scoring_program_ids = models.CompetitionPhase.objects.filter(competition=self.object).values_list('scoring_program_organizer_dataset')
        ingestion_program_ids = models.CompetitionPhase.objects.filter(competition=self.object).values_list('ingestion_program_organizer_dataset')

        public_data_organizer_dataset = models.OrganizerDataSet.objects.filter(
            Q(uploaded_by=self.request.user, type="Public Data") | Q(pk__in=public_data_ids)
        ).select_related('uploaded_by')
        starting_kit_organizer_dataset = models.OrganizerDataSet.objects.filter(
            Q(uploaded_by=self.request.user, type="Starting Kit") | Q(pk__in=starting_kit_ids)
        ).select_related('uploaded_by')
        input_data_organizer_dataset = models.OrganizerDataSet.objects.filter(
            Q(uploaded_by=self.request.user, type="Input Data") | Q(pk__in=input_data_ids)
        ).select_related('uploaded_by')
        reference_data_organizer_dataset = models.OrganizerDataSet.objects.filter(
            Q(uploaded_by=self.request.user, type="Reference Data") | Q(pk__in=reference_data_ids)
        ).select_related('uploaded_by')
        scoring_program_organizer_dataset = models.OrganizerDataSet.objects.filter(
            Q(uploaded_by=self.request.user, type="Scoring Program") | Q(pk__in=scoring_program_ids)
        ).select_related('uploaded_by')
        ingestion_program_organizer_dataset = models.OrganizerDataSet.objects.filter(
            Q(uploaded_by=self.request.user, type="Ingestion Program") | Q(pk__in=ingestion_program_ids)
        ).select_related('uploaded_by')

        for inline_form in inline_formsets[1].forms:
            inline_form.fields['public_data_organizer_dataset'].queryset = public_data_organizer_dataset
            inline_form.fields['starting_kit_organizer_dataset'].queryset = starting_kit_organizer_dataset
            inline_form.fields['input_data_organizer_dataset'].queryset = input_data_organizer_dataset
            inline_form.fields['reference_data_organizer_dataset'].queryset = reference_data_organizer_dataset
            inline_form.fields['scoring_program_organizer_dataset'].queryset = scoring_program_organizer_dataset
            inline_form.fields['ingestion_program_organizer_dataset'].queryset = ingestion_program_organizer_dataset
        return inline_formsets

    def get(self, request, *args, **kwargs):
        self.object = self.get_object()

        if self.object.creator != request.user and request.user not in self.object.admins.all():
            return HttpResponse(status=403)

        return super(CompetitionEdit, self).get(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        self.object = self.get_object()

        if self.object.creator != request.user and request.user not in self.object.admins.all():
            return HttpResponse(status=403)

        return super(CompetitionEdit, self).post(request, *args, **kwargs)


class CompetitionDelete(LoginRequiredMixin, DeleteView):
    """ View to Delete a competition."""
    model = models.Competition
    template_name = 'web/competitions/confirm-delete.html'
    success_url = '/my/#my_managing'

    def get(self, request, *args, **kwargs):
        self.object = self.get_object()

        if self.object.creator != request.user:
            return HttpResponse(status=403)

        context = self.get_context_data(object=self.object)
        return self.render_to_response(context)

    def delete(self, request, *args, **kwargs):
        self.object = self.get_object()

        if self.object.creator != request.user:
            return HttpResponse(status=403)

        success_url = self.get_success_url()
        self.object.delete()
        return HttpResponseRedirect(success_url)


@login_required
def competition_message_participants(request, competition_id):
    if request.method != "POST":
        return HttpResponse(status=400)

    try:
        competition = models.Competition.objects.get(pk=competition_id)
    except ObjectDoesNotExist:
        return HttpResponse(status=404)

    if competition.creator != request.user and request.user not in competition.admins.all():
        return HttpResponse(status=403)

    if "subject" not in request.POST and "body" not in request.POST:
        return HttpResponse(
            json.dumps({
                "error": "Missing subject or body of message!"
            }),
            status=400
        )

    participants = models.CompetitionParticipant.objects.filter(
        competition=competition,
        status=models.ParticipantStatus.objects.get(codename="approved"),
        user__organizer_direct_message_updates=True
    )
    emails = [p.user.email for p in participants]
    subject = request.POST.get('subject')
    body = strip_tags(request.POST.get('body'))

    if len(emails) > 0:
        tasks.send_mass_email.apply_async(
            (competition.pk,),
            dict(
                subject=subject,
                body=body,
                from_email=settings.DEFAULT_FROM_EMAIL,
                to_emails=emails
            )
        )

    return HttpResponse(status=200)


class UserDetailView(DetailView):
    """
    View to see User Details.
    """
    model = User
    template_name = 'web/user_detail.html'

    def get_context_data(self, **kwargs):
        context_data = super(UserDetailView, self).get_context_data(**kwargs)
        context_data['information'] = {
            'Organization': self.object.organization_or_affiliation,
            'Team Name': self.object.team_name,
            'Team Members': self.object.team_members,
            'Method Name': self.object.method_name,
            'Method Description': self.object.method_description,
            'Contact Email': self.object.contact_email,
            'Project URL': self.object.project_url,
            'Publication URL': self.object.publication_url,
            'Bibtex': self.object.bibtex,
        }
        return context_data


class CompetitionDetailView(DetailView):
    """Competiton Detail view."""
    queryset = models.Competition.objects.all()
    model = models.Competition
    template_name = 'web/competitions/view.html'

    def get(self, request, *args, **kwargs):
        competition = self.get_object()
        secret_key = request.GET.get("secret_key", None)
        if competition.creator != request.user and request.user not in competition.admins.all():
            # user may not be logged in, so grab PK if we can, to check if they are a participant
            user_pk = request.user.pk or -1
            if not competition.participants.filter(user=user_pk).exists():
                if not competition.published and competition.secret_key != secret_key:
                    return HttpResponse(status=404)
        # FIXME: handles legacy problem with missing post_save signal for forums, creates forum if it
        # does not exist for this competition. should be removed eventually.
        if not hasattr(competition, 'forum'):
            Forum.objects.get_or_create(competition=competition)
        return super(CompetitionDetailView, self).get(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super(CompetitionDetailView, self).get_context_data(**kwargs)
        competition = context['object']
        
        all_phases = competition.phases.all()

        # This assumes the tabs were created in the correct order
        # TODO Add a rank, order by on ContentCategory
        side_tabs = dict()
        for category in models.ContentCategory.objects.all():
            pagecontent = context['object'].pagecontent
            if pagecontent is not None:
                tc = [x for x in pagecontent.pages.filter(category=category)]
            else:
                tc = []
            side_tabs[category] = tc

        context['tabs'] = side_tabs
        context['site'] = Site.objects.get_current()
        context['current_server_time'] = datetime.now()

        context["first_phase"], context["previous_phase"], context['active_phase'], context["next_phase"] = get_first_previous_active_and_next_phases(competition)

        try:
            truncate_date = connection.ops.date_trunc_sql('day', 'submitted_at')
            score_def = SubmissionScoreDef.objects.filter(competition=competition).order_by('ordering').first()
            
            if score_def:
                if not score_def.computed:
                    qs = SubmissionScore.objects.filter(result__phase__competition=competition, scoredef=score_def)
                    qs = qs.extra({'day': truncate_date}).values('day')
                    if score_def.sorting == 'asc':
                        best_value = Max('value')
                    else:
                        best_value = Min('value')
                    qs = qs.annotate(high_score=best_value, count=Count('pk'))
                    #hhm add 2018.5.3
                # get submission_id and score for this competition
                chart1_data=[]
                qs_hhm = SubmissionScore.objects.filter(result__phase__competition=competition, scoredef=score_def)
                #get user_id in table web_competitionsubmission
                submission_id_list = qs_hhm.values("result","value")
                #user_id_list=[]
                for s in submission_id_list:
                    submission = models.CompetitionSubmission.objects.get(id=s['result'])
                    participant = ClUser.objects.filter(id=submission.participant_id)
                    user =models.CompetitionParticipant.objects.get(id=submission.participant_id)
                    user_name = ClUser.objects.get(id=user.user_id)
                    chart1_data.append({'user_name': user_name.username,'submission_id':s['result'],'score':s['value'],'day':submission.submitted_at})
                    
                context['graph'] = {
                    'days': [s['day'].strftime('%d %B %Y')  # ex 24 May 2017
                           for s in qs],
                    'high_scores': [s['high_score'] for s in qs],
                    'counts': [s['count'] for s in qs],
                    'sorting': score_def.sorting,
                    'chart1_data': chart1_data,
                }
                # Below is where we refactored top_three context.


            if context['active_phase']:
                try:
                    scores = context['active_phase'].scores()
                    headers = list(sorted(scores[0]['headers'], key=lambda x: x.get('ordering')))
                    default_score_key = headers[0]['key']

                    top_three_list = []

                    for group in scores:
                        for _, scoredata in group['scores']:
                            try:
                                default_score = next(val for val in scoredata['values'] if val['name'] == default_score_key)
                                top_three_list.append({
                                    "username": scoredata['username'],
                                    "score": default_score['val']
                                })
                            except (KeyError, StopIteration):
                                pass
                    context['top_three'] = top_three_list[0:3]
                except (KeyError, IndexError):
                    pass
        except ObjectDoesNotExist:
            context['top_three_leaders'] = None
            context['graph'] = None
            print("Could not find a score def!")

        if settings.USE_AWS:
            context['submission_upload_form'] = forms.SubmissionS3UploadForm

        submissions = dict()

        try:
            all_participants = competition.participants.all().select_related('user')

            if self.request.user.is_authenticated() and self.request.user in [x.user for x in all_participants]:
                context['my_status'] = [x.status for x in all_participants if x.user == self.request.user][0].codename
                context['my_participant'] = competition.participants.get(user=self.request.user)
                user_team = get_user_team(context['my_participant'], competition)
                context['my_team'] = user_team
                phase_iterator = iter(all_phases)
                for phase in phase_iterator:
                    submissions[phase] = models.CompetitionSubmission.objects.filter(participant=context['my_participant'], phase=phase)
                    if phase.is_active:
                        context['my_active_phase_submissions'] = submissions[phase]
                context['my_submissions'] = submissions

                # Add alert counters for admins
                context['comp_num_pending_teams'] = len(get_competition_pending_teams(competition))

                if user_team is None:
                    context['new_team_submission'] = 0
                    context['my_team_alert'] = 0

                    # Check if there are pending invitations from other teams
                    if len(get_user_requests(context['my_participant'], competition)) > 0:
                        context['my_team_alert'] = 1
                else:
                    context['new_team_submission'] = len(get_last_team_submissions(context['my_team'], 1))
                    if context['my_team'].creator == self.request.user:
                        context['my_team_alert'] = 0
                        # Check if my team has changed in the last 24 hours (accepted/edited)
                        if user_team.last_modified >= timezone.now() - timedelta(1):
                            context['my_team_alert'] = 1
                        else:
                            # Check if there are pending requests from other users
                            if len(get_team_pending_membership(user_team)) > 0:
                                context['my_team_alert'] = 1
                    else:
                        context['my_team_alert'] = 0
                        # TODO: Check if my membership changed in the last 24h
            else:
                context['my_status'] = "unknown"
                for phase in all_phases:
                    if phase.is_active:
                        context['active_phase'] = phase

        except ObjectDoesNotExist:
            pass

        if competition.creator == self.request.user or self.request.user in competition.admins.all():
            context['is_admin_or_owner'] = True

        # Use this flag to trigger container-fluid for result table
        context['on_competition_detail'] = True

        return context


class CompetitionSubmissionsPage(LoginRequiredMixin, TemplateView):
    """
    Serves the table of submissions in the Participate tab of a competition.

    .. note::

        Requires an authenticated user who is an approved participant of the competition.

    """
    template_name = 'web/competitions/_submit_results_page.html'

    def get_context_data(self, **kwargs):
        context = super(CompetitionSubmissionsPage, self).get_context_data(**kwargs)
        context['phase'] = None
        competition = models.Competition.objects.get(pk=self.kwargs['id'])

        if settings.USE_AWS:
            context['form'] = forms.SubmissionS3UploadForm

        if competition.participants.filter(user__in=[self.request.user]).exists():
            participant = competition.participants.get(user=self.request.user)
            if participant.status.codename == models.ParticipantStatus.APPROVED:
                phase = competition.phases.get(pk=self.kwargs['phase'])

                submissions = models.CompetitionSubmission.objects.filter(
                    participant=participant,
                    phase=phase
                ).select_related('status').order_by('submitted_at')

                # find which submission is in the leaderboard, if any and only if phase allows seeing results.
                id_of_submission_in_leaderboard = -1
                if not phase.is_blind:
                    leaderboard_entry = models.PhaseLeaderBoardEntry.objects.filter(
                        board__phase=phase,
                        result__participant__user=self.request.user
                    ).select_related('result', 'result__participant')
                    if leaderboard_entry:
                        id_of_submission_in_leaderboard = leaderboard_entry[0].result.pk
                submission_info_list = []
                for submission in submissions:
                    try:
                        default_score = float(submission.get_default_score())
                    except (TypeError, ValueError):
                        default_score = '---'
                    submission_info = {
                        'id': submission.id,
                        'number': submission.submission_number,
                        'filename': submission.get_filename(),  # left as call for legacy update of readable_filename on subs.
                        'submitted_at': submission.submitted_at,
                        'status_name': submission.status.name,
                        'is_finished': submission.status.codename == 'finished',
                        'is_in_leaderboard': submission.id == id_of_submission_in_leaderboard,
                        'exception_details': submission.exception_details,
                        'description': submission.description,
                        'team_name': submission.team_name, 
                        'method_name': submission.method_name,
                        'method_description': submission.method_description,
                        'project_url': submission.project_url,
                        'publication_url': submission.publication_url,
                        'bibtex': submission.bibtex,
                        'organization_or_affiliation': submission.organization_or_affiliation,
                        'is_public': submission.is_public,
                        'score': default_score,
                    }
                    submission_info_list.append(submission_info)
                context['submission_info_list'] = submission_info_list
                context['phase'] = phase

        try:
            last_submission = models.CompetitionSubmission.objects.filter(
                participant__user=self.request.user,
                phase=context['phase']
            ).latest('submitted_at')
            context['last_submission_team_name'] = last_submission.team_name
            context['last_submission_method_name'] = last_submission.method_name
            context['last_submission_method_description'] = last_submission.method_description
            context['last_submission_project_url'] = last_submission.project_url
            context['last_submission_publication_url'] = last_submission.publication_url
            context['last_submission_bibtex'] = last_submission.bibtex
            context['last_submission_organization_or_affiliation'] = last_submission.organization_or_affiliation
        except ObjectDoesNotExist:
            pass
        return context


@login_required()
def competition_submission_metadata_page(request, competition_id, phase_id):
    try:
        competition = models.Competition.objects.get(pk=competition_id)
        selected_phase = competition.phases.all().prefetch_related(
            'submissions',
            'submissions__status',
            'submissions__participant',
            'submissions__metadatas',
        ).get(pk=phase_id)
    except ObjectDoesNotExist:
        raise Http404()

    if request.user.id != competition.creator_id and request.user not in competition.admins.all():
        raise Http404()

    # Page does not display subs without these statuses
    subs = selected_phase.submissions.filter(status__name__in=['Finished', 'Failed']).order_by('submitted_at')

    paginator = Paginator(subs, 25)
    page = request.GET.get('page', 1)

    submission_list = paginator.page(page)

    return render(request, "web/competitions/submission_metadata.html", {
        'competition': competition,
        'selected_phase': selected_phase,
        'submission_list': submission_list,
        'stretch_100_percent_width': True
    })


class CompetitionResultsPage(TemplateView):
    """Serves the leaderboards in the Results tab of a competition."""
    template_name = 'web/competitions/_results_page.html'

    def get_context_data(self, **kwargs):
        context = super(CompetitionResultsPage, self).get_context_data(**kwargs)
        try:
            context['block_leaderboard_view'] = True
            competition = models.Competition.objects.get(pk=self.kwargs['id'])
            
            # hhm add 2018.5.4
            score_def = SubmissionScoreDef.objects.filter(competition=competition).order_by('ordering').first()
            
            phase = competition.phases.get(pk=self.kwargs['phase'])
            is_owner = self.request.user.id == competition.creator_id
            context['competition_admins'] = competition.admins.all()
            context['is_owner'] = is_owner
            context['phase'] = phase
            context['groups'] = phase.scores()

            for group in context['groups']:
                for _, scoredata in group['scores']:
                    sub = models.CompetitionSubmission.objects.get(pk=scoredata['id'])
                    scoredata['date'] = sub.submitted_at
                    scoredata['count'] = sub.phase.submissions.filter(participant=sub.participant).count()
                    if sub.team:
                        scoredata['team_name'] = sub.team.name

            user = self.request.user

            # Will allow creator and admin to see Leaderboard in advanced
            if user == phase.competition.creator or user in phase.competition.admins.all():
                context['block_leaderboard_view'] = False

            #hhm add 2018.5.3
            # get submission_id and score for this competition
            chart1_data=[]
            qs_hhm = SubmissionScore.objects.filter(result__phase__competition=competition, scoredef=score_def)
            #get user_id in table web_competitionsubmission
            submission_id_list = qs_hhm.values("result","value")
            #user_id_list=[]
            for s in submission_id_list:
                submission = models.CompetitionSubmission.objects.get(id=s['result'])
                participant = ClUser.objects.filter(id=submission.participant_id)
                
                user =models.CompetitionParticipant.objects.get(id=submission.participant_id)
                
                user_name = ClUser.objects.get(id=user.user_id)
                
                chart1_data.append({'user_name': user_name.username,'submission_id':s['result'],'score':s['value'],'day':submission.submitted_at})
                #user_id_list.append(submission.participant_id)
            
            context['chart1_data'] = chart1_data
            return context
        except:
            context['error'] = traceback.format_exc()
            return context


class CompetitionPublicSubmission(TemplateView):
    ''' Returns the public  submissions of a competition.'''
    template_name = 'web/competitions/public_submissions.html'

    def get_context_data(self, **kwargs):
        context = super(CompetitionPublicSubmission, self).get_context_data(**kwargs)
        context['active_phase'] = None
        competition = None

        try:
            competition = models.Competition.objects.get(pk=self.kwargs['pk'])
            context['competition'] = competition

            for phase in competition.phases.all():
                if phase.is_active:
                    context['active_phase'] = phase
        except:
            context['error'] = traceback.print_exc()

        # In case all phases are close, lets get last phase
        if context['active_phase'] is None and competition:
            context['active_phase'] = competition.phases.all().order_by("phasenumber").reverse()[0]
        return context


class CompetitionPublicSubmissionByPhases(TemplateView):
    '''Returns the submissions of a specific phase for a competition.

    .. note::

        - We are using a Ajax request for this. Look into 'public_submissions.html' for more info
        - Results will append to '_public_submissions_phases.html'. Look at the file for more info
    '''
    template_name = 'web/competitions/public_submissions_phase.html'

    def get_context_data(self, **kwargs):

        context = super(CompetitionPublicSubmissionByPhases, self).get_context_data(**kwargs)
        try:
            competition = models.Competition.objects.get(pk=self.kwargs['pk'])
            competition_phase = self.kwargs['phase']
            context['competition'] = competition
            context['public_submissions'] = []
            public_submissions = models.CompetitionSubmission.objects.filter(phase__competition=competition,
                                                                             phase__pk = competition_phase,
                                                                             is_public=True,
                                                                             status__codename="finished").select_related('participant__user').prefetch_related('phase')

            for submission in public_submissions:
                # Let's process all public submissions and figure out which ones we've already liked

                if self.request.user.is_authenticated():
                    if Like.objects.filter(submission=submission, user=self.request.user).exists():
                        submission.already_liked = True
                    if Dislike.objects.filter(submission=submission, user=self.request.user).exists():
                        submission.already_disliked = True
                context['public_submissions'].append(submission)
        except:
            context['error'] = traceback.print_exc()
        return context


class CompetitionResultsDownload(View):
    """View to download the results of a competition."""

    def get(self, request, *args, **kwargs):
        competition = models.Competition.objects.get(pk=self.kwargs['id'])
        phase = competition.phases.get(pk=self.kwargs['phase'])
        response = HttpResponse(competition.get_results_csv(phase.pk, request=request), status=200, content_type="text/csv")
        my_response = ("attachment; filename=%s results.csv" % phase.competition.title).encode('ascii', 'ignore').strip()
        response["Content-Disposition"] = my_response
        return response


class CompetitionCompleteResultsDownload(View):
    """Views to download a complete version of competitions results."""

    def get(self, request, *args, **kwargs):
        competition = models.Competition.objects.get(pk=self.kwargs['id'])
        phase = competition.phases.get(pk=self.kwargs['phase'])

        groups = phase.scores(include_scores_not_on_leaderboard=True)
        leader_board = models.PhaseLeaderBoard.objects.get(phase=phase)

        csvfile = StringIO.StringIO()
        csvwriter = csv.writer(csvfile)

        for group in groups:
            csvwriter.writerow([group['label'].encode("utf-8")])
            csvwriter.writerow([])

            headers = ["User"]
            sub_headers = [""]
            for header in group['headers']:
                subs = header['subs']
                if subs:
                    for sub in subs:
                        headers.append(header['label'].encode("utf-8"))
                        sub_headers.append(sub['label'].encode("utf-8"))
                else:
                    headers.append(header['label'].encode("utf-8"))
            headers.append('Description')
            headers.append('Date')
            headers.append('Filename')
            headers.append('Is on leaderboard?')
            csvwriter.writerow(headers)
            csvwriter.writerow(sub_headers)

            if len(group['scores']) <= 0:
                csvwriter.writerow(["No data available"])
            else:
                leader_board_entries = models.PhaseLeaderBoardEntry.objects.filter(board=leader_board).values_list('result__id', flat=True)

                for pk, scores in group['scores']:
                    submission = models.CompetitionSubmission.objects.get(pk=scores['id'])
                    row = [scores['username']]
                    for v in scores['values']:
                        if 'rnk' in v:
                            row.append("%s (%s)" % (v['val'], v['rnk']))
                        else:
                            row.append("%s (%s)" % (v['val'], v['hidden_rnk']))

                    row.append(submission.description)
                    row.append(submission.submitted_at)
                    row.append(submission.get_filename())

                    is_on_leaderboard = submission.pk in leader_board_entries
                    row.append(is_on_leaderboard)

                    row = [unicode(r).encode("utf-8") for r in row]
                    csvwriter.writerow(row)

            csvwriter.writerow([])
            csvwriter.writerow([])

        response = HttpResponse(csvfile.getvalue(), status=200, content_type="text/csv")
        response["Content-Disposition"] = "attachment; filename=competition_results.csv"
        return response


# Views for My Codalab

class MyIndex(LoginRequiredMixin):
    pass


class MyCompetitionParticipantView(LoginRequiredMixin, ListView):
    """View that returns all participants from a competition."""
    queryset = models.CompetitionParticipant.objects.all()
    template_name = 'web/my/participants.html'

    def get_context_data(self, **kwargs):
        context = super(MyCompetitionParticipantView, self).get_context_data(**kwargs)
        # create column definition
        columns = [
            {
                'label': '#',
                'name': 'number'
            },
            {
                'label': 'NAME',
                'name': 'name'
            },
            {
                'label': 'EMAIL',
                'name': 'email'
            },
            {
                'label': 'TEAM',
                'name': 'team_name'
            },
            {
                'label': 'STATUS',
                'name': 'status'
            },
            {
                'label': 'ENTRIES',
                'name': 'entries'
            }
        ]

        team_columns = [
            {
                'label': '#',
                'name': 'number'
            },
            {
                'label': 'NAME',
                'name': 'name'
            },
            {
                'label': 'CREATOR',
                'name': 'creator'
            },
            {
                'label': '# MEMBERS',
                'name': 'num_members'
            },
            {
                'label': 'PENDING REQ/INV',
                'name': 'num_pending'
            },
            {
                'label': 'STATUS',
                'name': 'status'
            },
            {
                'label': 'ENTRIES',
                'name': 'entries'
            }
        ]

        try:
            competition = models.Competition.objects.get(pk=self.kwargs.get('competition_id'))
        except models.Competition.DoesNotExist:
            raise Http404()

        if competition.creator != self.request.user and self.request.user not in competition.admins.all():
            raise Http404()

        context['columns'] = columns
        context['team_columns'] = team_columns
        # retrieve participant submissions information
        participant_list = []
        competition_participants = self.queryset.filter(competition=competition).order_by('pk')
        competition_participants_ids = list(participant.id for participant in competition_participants)
        context['pending_participants'] = filter(lambda participant_submission: participant_submission.status.codename == models.ParticipantStatus.PENDING, competition_participants)
        participant_submissions = models.CompetitionSubmission.objects.filter(participant__in=competition_participants_ids)
        for number, participant in enumerate(competition_participants):
            team = get_user_team(participant.user, participant.competition)
            if team is not None:
                team_name = team.name
            else:
                team_name = ''
            participant_entry = {
                'pk': participant.pk,
                'name': participant.user.username,
                'email': participant.user.email,
                'user_pk': participant.user.pk,
                'status': participant.status.codename,
                'number': number + 1,
                # equivalent to assigning participant.submissions.count() but without several multiple db queires
                'entries': len(filter(lambda participant_submission: participant_submission.participant.id == participant.id, participant_submissions)),
                'team_name': team_name,
                'team': team
            }
            participant_list.append(participant_entry)
        # order results
        sort_data_table(self.request, context, participant_list)
        context['participant_list'] = participant_list
        context['competition_id'] = self.kwargs.get('competition_id')

        # If teams are enabled for this competition, add team information
        competition = models.Competition.objects.get(pk=self.kwargs.get('competition_id'))
        context['allow_organizer_teams'] = competition.allow_organizer_teams
        context['csv_form'] = OrganizerTeamsCSVForm(competition_pk=self.kwargs.get('competition_id'), creator_pk=competition.creator.pk)
        if competition.enable_teams or competition.allow_organizer_teams:
            if competition.enable_teams:
                context['teams_enabled'] = True
            participant_memberships = TeamMembership.objects.filter(user__in=competition_participants_ids)
            teams_list = []
            for number, team in enumerate(get_competition_teams(competition)):
                team_entry = {
                    'pk': team.pk,
                    'name': team.name,
                    'creator': team.creator.username,
                    'creator_pk': team.creator.pk,
                    'num_members': team.members.count(),
                    'num_pending': 0,
                    'status': team.status.codename,
                    'number': number + 1,
                    # equivalent to assigning participant.submissions.count() but without several multiple db queires
                    'entries': len(filter(
                        lambda participant_submission: get_user_team(participant_submission.participant,
                                                                     competition) == team, participant_submissions)),
                }
                teams_list.append(team_entry)
            context['team_list'] = teams_list
            context['pending_teams'] = get_competition_pending_teams(competition)
        return context

    def get_queryset(self):
        return self.queryset.filter(competition=self.kwargs.get('competition_id'))


# Partials

class CompetitionIndexPartial(TemplateView):

    def get_context_data(self, **kwargs):
        ## Currently gets all competitions
        context = super(CompetitionIndexPartial, self).get_context_data(**kwargs)
        per_page = self.request.GET.get('per_page', 6)
        page = self.request.GET.get('page', 1)
        clist = models.Competition.objects.all()

        pgn = Paginator(clist, per_page)
        try:
            competitions = pgn.page(page)
        except PageNotAnInteger:
            # If page is not an integer, deliver first page.
            competitions = pgn.page(1)
        except EmptyPage:
            # If page is out of range (e.g. 9999), deliver last page of results.
            competitions = []
        context['competitions'] = competitions
        return context


class MyCompetitionsManagedPartial(ListView):
    model = models.Competition
    template_name = 'web/my/_managed.html'
    queryset = models.Competition.objects.all()

    def get_queryset(self):
        return self.queryset.filter(creator=self.request.user)


class MyCompetitionsEnteredPartial(ListView):
    model = models.CompetitionParticipant
    template_name = 'web/my/_entered.html'
    queryset = models.CompetitionParticipant.objects.all()

    def get_queryset(self):
        return self.queryset.filter(user=self.request.user)


class MyCompetitionDetailsTab(TemplateView):
    template_name = 'web/my/_tab.html'


class MySubmissionResultsPartial(TemplateView):
    template_name = 'web/my/_submission_results.html'

    def get_context_data(self, **kwargs):
        ctx = super(MySubmissionResultsPartial, self).get_context_data(**kwargs)

        participant_id = kwargs.get('participant_id')
        participant = models.CompetitionParticipant.objects.get(pk=participant_id)

        phase_id = kwargs.get('phase_id')
        phase = models.CompetitionPhase.objects.get(pk=phase_id)

        ctx['active_phase'] = phase
        ctx['my_active_phase_submissions'] = phase.submissions.filter(participant=participant)

        return ctx


class MyCompetitionSubmissionToggleMakePublic(LoginRequiredMixin, View):
    """
    Makes a submission public.

    .. note:

        Admins, creator and submission's owner are able to published a submission.
    """
    def get(self, request, *args, **kwargs):
        try:
            submission = models.CompetitionSubmission.objects.get(pk=kwargs.get('submission_id'))
            if request.user == submission.participant.user or request.user == submission.phase.competition.creator \
                or request.user in submission.phase.competition.admins.all():
                submission.is_public = not submission.is_public
                submission.save()
                return HttpResponse(submission.is_public)
            else:
                raise Http404()
        except ObjectDoesNotExist:
            raise Http404()


class MyCompetitionSubmissionOutput(View):
    """
    This view serves the files associated with a submission.
    """
    def get(self, request, *args, **kwargs):
        try:
            submission = models.CompetitionSubmission.objects.get(pk=kwargs.get('submission_id'))
        except ObjectDoesNotExist:
            raise Http404()
        competition = submission.phase.competition
        filetype = kwargs.get('filetype')

        # Check competition admin permissions or user permissions
        if filetype == "detailed_results.html":
            published_to_leaderboard = models.PhaseLeaderBoardEntry.objects.filter(result=submission).exists()
        else:
            published_to_leaderboard = False

        if not submission.is_public and not published_to_leaderboard:
            if (competition.creator != request.user and request.user not in competition.admins.all()) and \
                            request.user != submission.participant.user:
                raise Http404()

        try:
            file, file_type, file_name = submission.get_file_for_download(
                filetype,
                request.user,
                override_permissions=published_to_leaderboard
            )
        except PermissionDenied:
            return HttpResponse(status=403)
        except ValueError:
            return HttpResponse(status=400)
        except:
            return HttpResponse(status=500)
        if settings.USE_AWS:
            if file_name:
                return HttpResponseRedirect(
                    _make_url_sassy(file_name)
                )
            else:
                raise Http404()
        else:
            try:
                response = HttpResponse(file.read(), status=200, content_type=file_type)
                if file_type == 'application/zip':
                    response['Content-Type'] = 'application/zip'
                    response['Content-Disposition'] = 'attachment; filename="{0}"'.format(file_name)
                else:
                    response['Content-Type'] = file_type
                return response
            except azure.WindowsAzureMissingResourceError:
                # for stderr.txt which does not exist when no errors have occurred
                # this may hide a true 404 in unexpected circumstances
                return HttpResponse("", status=200, content_type='text/plain')
            except:
                # Let's check to make sure we're in a prediction competition, otherwise let user know
                if filetype.startswith("predict_") and submission.phase.is_scoring_only:
                    return HttpResponse("This competition is scoring only, prediction data not available",
                                        content_type='text/plain')
                else:
                    msg = "There was an error retrieving file '%s'. Please try again later or report the issue."
                    return HttpResponse(msg % filetype, status=200, content_type='text/plain')


class MyCompetitionSubmissionDetailedResults(TemplateView):
    """
    This view serves the files associated with a submission.
    """
    model = models.CompetitionSubmission
    template_name = 'web/my/detailed_results.html'

    def get(self, request, *args, **kwargs):
        submission = models.CompetitionSubmission.objects.get(pk=kwargs.get('submission_id'))

        context_dict = {
            'id': kwargs.get('submission_id'),
            'user': submission.participant.user,
            'submission': submission,
        }
        return render_to_response('web/my/detailed_results.html', context_dict, RequestContext(request))


class MyCompetitionSubmissionsPage(LoginRequiredMixin, TemplateView):
    """Serves the table of submissions in the submissions competition administration.

    .. note::

        Requires an authenticated user who is an administrator of the competition."""
    queryset = models.Competition.objects.all()
    model = models.Competition
    template_name = 'web/my/submissions.html'

    def get_context_data(self, **kwargs):
        phase_id = self.request.GET.get('phase')
        context = super(MyCompetitionSubmissionsPage, self).get_context_data(**kwargs)
        competition = models.Competition.objects.get(pk=self.kwargs['competition_id'])
        context['competition'] = competition

        if self.request.user.id != competition.creator_id and self.request.user not in competition.admins.all():
            raise Http404()

        # find the active phase
        if phase_id:
            context['selected_phase_id'] = int(phase_id)
            active_phase = competition.phases.filter(id=phase_id)[0]
        else:
            phases = list(competition.phases.all())
            active_phase = phases[0]
            for phase in phases:
                if phase.is_active:
                    context['selected_phase_id'] = phase.id
                    active_phase = phase

        context['selected_phase'] = active_phase

        submissions = models.CompetitionSubmission.objects.filter(phase=active_phase).select_related('participant', 'participant__user', 'status')
        # find which submissions are in the leaderboard, if any and only if phase allows seeing results.
        leaderboard_entries = list(models.PhaseLeaderBoardEntry.objects.filter(board__phase__competition=competition))
        id_of_submissions_in_leaderboard = [e.result.id for e in leaderboard_entries if e.result in submissions]
        # create column definition
        columns = [
            {
                'label': 'SUBMITTED',
                'name': 'submitted_at'
            },
            {
                'label': 'SUBMITTED BY',
                'name': 'submitted_by'
            },
            {
                'label': 'SUBMISSION ID',
                'name': 'submission_pk'
            },
            {
                'label': 'FILENAME',
                'name': 'filename'
            },
            {
                'label': 'STATUS',
                'name': 'status_name'
            },
            {
                'label': 'LEADERBOARD',
                'name': 'is_in_leaderboard'
            },
        ]

        # This line is causing problems...
        # Active phase should be fine
        scores = active_phase.scores(include_scores_not_on_leaderboard=True)
        bad_score_count, bad_scores = check_bad_scores(scores)
        try:
            if bad_score_count > 0:
                raise ScoringException("Improperly configured leaderboard or scoring program!")
        except ScoringException:
            context['scoring_exception'] = "ERROR: Improperly configured leaderboard or scoring program. Some " \
                                           "scores have NaN! Please check your leaderboard configuration and" \
                                           " scoring program for the competition!"
            context['bad_scores'] = bad_scores
            context['bad_score_count'] = bad_score_count

        for score_group_index, score_group in enumerate(scores):
            column = {
                'label': score_group['label'],
                'name': 'score_' + str(score_group_index),
            }
            columns.append(column)
        # map submissions to view data
        submission_info_list = []
        for submission in submissions:
            submission_info = {
                'id': submission.id,
                'submitted_by': submission.participant.user.username,
                'user_pk': submission.participant.user.pk,
                'number': submission.submission_number,
                'filename': submission.get_filename(),
                'submitted_at': submission.submitted_at,
                'status_name': submission.status.name,
                'is_in_leaderboard': submission.id in id_of_submissions_in_leaderboard,
                'exception_details': submission.exception_details,
                'description': submission.description,
                'is_public': submission.is_public,
                'submission_pk': submission.id,
                'is_migrated': submission.is_migrated
            }
            # Removed if to show scores on submissions view.

            #if (submission_info['is_in_leaderboard'] == True):
            # add score groups into data columns
            for score_group_index, score_group in enumerate(scores):
                # Need to figure out a way to check if submission is garbage.
                try:
                    user_score = filter(lambda user_score: user_score[1]['id'] == submission.id, score_group['scores'])[0] # This line return error.
                    main_score = filter(lambda main_score: main_score['name'] == score_group['headers'][0]['key'], user_score[1]['values'])[0]
                    submission_info['score_' + str(score_group_index)] = main_score['val']
                # If submission is garbage put in garbage data.
                except:
                    user_score = "---"
                    main_score = "---"
                    submission_info['score_' + str(score_group_index)] = "---"
            submission_info_list.append(submission_info)
        # order results
        sort_data_table(self.request, context, submission_info_list)
        # complete context
        context['columns'] = columns

        paginator = Paginator(submission_info_list, 25)
        page = self.request.GET.get('page')
        if page is None:
            # If we don't get page back in the request dict, default to 1st page.
            page = 1

        context['submission_info_list'] = paginator.page(page)

        # We need a way to check if next phase.auto_migration = True
        try:
            next_phase = competition.phases.get(phasenumber=submission.phase.phasenumber+1)
            context['next_phase'] = next_phase.auto_migration
        except Exception:
            sys.exc_clear()
        context['phase'] = active_phase

        if competition.creator == self.request.user or self.request.user in competition.admins.all():
            context['is_admin_or_owner'] = True

        return context


class VersionView(TemplateView):
    template_name = 'web/project_version.html'

    def get_context_data(self):
        import subprocess
        p = subprocess.Popen(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE)
        out, err = p.communicate()
        ctx = super(VersionView, self).get_context_data()
        ctx['commit_hash'] = out
        tasks.echo("version is " + out)
        return ctx


class OrganizerDataSetListView(LoginRequiredMixin, ListView):
    model = models.OrganizerDataSet
    template_name = "web/my/datasets.html"

    def get_queryset(self):
        return models.OrganizerDataSet.objects.filter(uploaded_by=self.request.user)


class OrganizerDataSetFormMixin(LoginRequiredMixin):
    model = models.OrganizerDataSet
    form_class = forms.OrganizerDataSetModelForm
    template_name = "web/my/datasets_form.html"

    def get_form_kwargs(self, **kwargs):
        kwargs = super(OrganizerDataSetFormMixin, self).get_form_kwargs(**kwargs)
        kwargs['user'] = self.request.user
        return kwargs

    def get_form(self, form_class):
        form = super(OrganizerDataSetFormMixin, self).get_form(form_class)
        form.fields["sub_data_files"].queryset = models.OrganizerDataSet.objects.filter(
            uploaded_by=self.request.user,
            sub_data_files__isnull=True, # ignore datasets that are multi
        )
        return form

    def get_success_url(self):
        return reverse("my_datasets")


class OrganizerDataSetCreate(OrganizerDataSetFormMixin, CreateView):
    model = models.OrganizerDataSet
    form_class = forms.OrganizerDataSetModelForm
    template_name = "web/my/datasets_form.html"

    def get_form_kwargs(self, **kwargs):
        kwargs = super(OrganizerDataSetCreate, self).get_form_kwargs(**kwargs)
        kwargs['user'] = self.request.user
        return kwargs

    def get_success_url(self):
        return reverse("my_datasets")


class OrganizerDataSetCheckOwnershipMixin(LoginRequiredMixin):
    def get_object(self, queryset=None):
        dataset = super(OrganizerDataSetCheckOwnershipMixin, self).get_object(queryset)

        if dataset.uploaded_by != self.request.user:
            raise Http404()

        return dataset


class OrganizerDataSetUpdate(OrganizerDataSetCheckOwnershipMixin, OrganizerDataSetFormMixin, UpdateView):
    pass


class OrganizerDataSetDelete(OrganizerDataSetCheckOwnershipMixin, DeleteView):
    model = models.OrganizerDataSet
    template_name = "web/my/datasets_delete.html"

    def get_success_url(self):
        return reverse("my_datasets")

    def get_context_data(self, **kwargs):
        context = super(OrganizerDataSetDelete, self).get_context_data(**kwargs)

        usage = models.Competition.objects.all()

        if self.object.type == "Input Data":
            usage = usage.filter(phases__input_data_organizer_dataset=self.object)
        elif self.object.type == "Reference Data":
            usage = usage.filter(phases__reference_data_organizer_dataset=self.object)
        elif self.object.type == "Scoring Program":
            usage = usage.filter(phases__scoring_program_organizer_dataset=self.object)
        else:
            usage = usage.filter(Q(phases__input_data_organizer_dataset=self.object) |
                                 Q(phases__reference_data_organizer_dataset=self.object) |
                                 Q(phases__scoring_program_organizer_dataset=self.object))

        # Filter out duplicates
        context["competitions_in_use"] = usage.distinct()
        return context


class SubmissionDelete(LoginRequiredMixin, DeleteView):
    model = models.CompetitionSubmission
    template_name = "web/my/submission_delete.html"

    def get_object(self, queryset=None):
        obj = super(SubmissionDelete, self).get_object(queryset)

        is_admin = self.request.user in obj.phase.competition.admins.all()
        # Check user is owner, competition creator, or competition admin
        if obj.participant.user != self.request.user and obj.phase.competition.creator != self.request.user and not is_admin:
            raise Http404()
        self.success_url = reverse("competitions:view", kwargs={"pk": obj.phase.competition.pk})
        return obj

def download_dataset(request, dataset_key):
    """
    Downloads a dataset that belongs to authenticated user

    :param dataset_key: Primary key for dataset
    """
    try:
        dataset = models.OrganizerDataSet.objects.get(key=dataset_key)
    except ObjectDoesNotExist:
        raise Http404()

    try:
        if dataset.sub_data_files.count() > 0:
            # TODO: Could refactor this to only zip this stuff up one time, maybe after dataset creation?
            zip_buffer = StringIO.StringIO()

            zip_file = zipfile.ZipFile(zip_buffer, "w")
            file_name = ""

            for sub_dataset in dataset.sub_data_files.all():
                file_dir, file_name = os.path.split(sub_dataset.data_file.file.name)
                zip_file.writestr(file_name, sub_dataset.data_file.read())

            zip_file.close()

            resp = HttpResponse(zip_buffer.getvalue(), mimetype = "application/x-zip-compressed")
            resp['Content-Disposition'] = 'attachment; filename=%s.zip' % dataset.name
            return resp
        else:
            return HttpResponseRedirect(_make_url_sassy(dataset.data_file.file.name))
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print "*** print_tb:"
        traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
        print "*** print_exception:"
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  limit=2, file=sys.stdout)
        print "*** print_exc:"
        traceback.print_exc()
        print "*** format_exc, first and last line:"
        formatted_lines = traceback.format_exc().splitlines()
        msg = "There was an error retrieving the file. Please try again later or report the issue."
        return HttpResponse(msg, status=400, content_type='text/plain')


def datasets_delete_multiple(request):
    """
    Deletes multiple datasets.

    :param id: ids of datasets to be deleted
    """
    ids_to_delete = request.POST.getlist("ids_to_delete[]", [])

    for dataset_id in ids_to_delete:
        try:
            dataset = models.OrganizerDataSet.objects.get(pk=int(dataset_id), uploaded_by=request.user)
            dataset.delete()
        except:
            pass

    return HttpResponse()


def download_competition_yaml(request, competition_pk):
    """
    Downloads competition yaml file.

    :param competition_pk: Competition primary key.

    .. note::

        User needs to be creator of admin of competition.

    """
    try:
        competition = models.Competition.objects.get(pk=competition_pk)

        if competition.creator != request.user and request.user not in competition.admins.all():
            return HttpResponse(status=403)

        response = HttpResponse(competition.original_yaml_file, content_type="text/yaml")
        response['Content-Disposition'] = 'attachment; filename="competition_%s.yaml"' % competition_pk
        return response
    except ObjectDoesNotExist:
        return HttpResponse(status=404)


@login_required
def download_competition_bundle(request, competition_pk):
    """
    Downloads competition bundle.

    :param competition_pk: Competition's primary key.

    .. note::

        User needs to be creator of admin of competition.
    """
    if not request.user.is_staff:
        return HttpResponse(status=403)

    try:
        competition = models.Competition.objects.get(pk=competition_pk)
    except ObjectDoesNotExist:
        raise Http404()

    try:
        zip_buffer = StringIO.StringIO()
        zip_file = zipfile.ZipFile(zip_buffer, "w")
        yaml_data = yaml.load(competition.original_yaml_file)

        # Grab logo
        zip_file.writestr(yaml_data["image"], competition.image.file.read())

        # Grab html pages
        for p in competition.pagecontent.pages.all():
            if p.codename in yaml_data["html"].keys() or p.codename == 'terms_and_conditions' or p.codename == 'get_data':
                if p.codename == 'terms_and_conditions':
                    # overwrite this for consistency
                    p.codename = 'terms'
                if p.codename == 'get_data':
                    # overwrite for consistency
                    p.codename = 'data'
                zip_file.writestr(yaml_data["html"][p.codename], p.html.encode("utf-8"))

        # Grab input data, reference data, scoring program
        file_name_cache = []

        for phase in competition.phases.all():
            for phase_index, phase_yaml in yaml_data["phases"].items():
                if phase_yaml["phasenumber"] == phase.phasenumber:
                    if phase.reference_data and phase.reference_data.file.name not in file_name_cache:
                        yaml_data["phases"][phase_index]["reference_data"] = phase.reference_data.file.name
                        file_name_cache += phase.reference_data.file.name
                        zip_file.writestr(phase.reference_data.file.name, phase.reference_data.file.read())

                    if phase.input_data and phase.input_data.file.name not in file_name_cache:
                        yaml_data["phases"][phase_index]["input_data"] = phase.input_data.file.name
                        file_name_cache += phase.input_data.file.name
                        zip_file.writestr(phase.input_data.file.name, phase.input_data.file.read())

                    if phase.scoring_program and phase.scoring_program.file.name not in file_name_cache:
                        yaml_data["phases"][phase_index]["scoring_program"] = phase.scoring_program.file.name
                        file_name_cache += phase.scoring_program.file.name
                        zip_file.writestr(phase.scoring_program.file.name, phase.scoring_program.file.read())

        zip_file.writestr("competition.yaml", yaml.dump(yaml_data))

        zip_file.close()

        resp = HttpResponse(zip_buffer.getvalue(), mimetype = "application/x-zip-compressed")
        resp['Content-Disposition'] = 'attachment; filename=%s-%s.zip' % (competition.title, competition.pk)
        return resp
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print "*** print_tb:"
        traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
        print "*** print_exception:"
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  limit=2, file=sys.stdout)
        print "*** print_exc:"
        traceback.print_exc()
        print "*** format_exc, first and last line:"
        formatted_lines = traceback.format_exc().splitlines()
        msg = "There was an error retrieving the file. Please try again later or report the issue."
        return HttpResponse(msg, status=400, content_type='text/plain')


@login_required
def download_leaderboard_results(request, competition_pk, phase_pk):
    """
    Downloads submissions from the leaderboard table.

    :param competition_pk: Competition's primary key
    :param phase_pk: Phase's primary key
    """
    try:
        competition = models.Competition.objects.get(pk=competition_pk)
        if competition.creator != request.user and request.user not in competition.admins.all():
            raise Http404()

        phase = models.CompetitionPhase.objects.get(pk=phase_pk)
        leaderboard_entries = models.PhaseLeaderBoardEntry.objects.filter(board__phase=phase)
    except ObjectDoesNotExist:
        raise Http404()

    try:
        zip_buffer = StringIO.StringIO()
        zip_file = zipfile.ZipFile(zip_buffer, "w")

        # Add teach team name in an easy to read way
        team_name_cache = {}
        team_name_string = ""
        for result in models.PhaseLeaderBoardEntry.objects.filter(result__participant__user__team_name__isnull=False,
                                                                  result__participant__competition=competition):
            user_on_team = result.result.participant.user
            team_name_cache[user_on_team.team_name] = user_on_team.team_members
        for name, members in team_name_cache.items():
            team_name_string += "Team: %s; members: %s\n" % (name, members)

        if team_name_string:
            zip_file.writestr("team_names_and_members.txt", team_name_string.encode('utf8'))

        # Add each submission
        for entry in leaderboard_entries:
            # Maps back to submission
            submission = entry.result
            username_or_team_name = submission.participant.user.username if not submission.participant.user.team_name else "Team %s " % submission.participant.user.team_name
            file_name = "%s - %s submission.zip" % (username_or_team_name, submission.submission_number)

            if settings.USE_AWS:
                url = _make_url_sassy(submission.s3_file)
                zip_file.writestr(file_name, urllib.urlopen(url).read())
            else:
                zip_file.writestr(file_name, submission.file.read())

            output_file_name = "%s - %s output.zip" % (username_or_team_name, submission.submission_number)
            zip_file.writestr(output_file_name, submission.output_file.read())

            profile_data_file_name = "%s - %s profile.txt" % (username_or_team_name, submission.submission_number)
            user_profile_data = {
                'Organization': submission.participant.user.organization_or_affiliation,
                'Team Name': submission.participant.user.team_name,
                'Team Members': submission.participant.user.team_members,
                'Method Name': submission.participant.user.method_name,
                'Method Description': submission.participant.user.method_description,
                'Contact Email': submission.participant.user.contact_email,
                'Project URL': submission.participant.user.project_url,
                'Publication URL': submission.participant.user.publication_url,
                'Bibtex': submission.participant.user.bibtex,
            }
            user_profile_data_string = '\n'.join(['%s: %s' % (k, v) for k, v in user_profile_data.items()])
            zip_file.writestr(profile_data_file_name, user_profile_data_string.encode('utf-8'))

            metadata_fields = ['method_name', 'method_description', 'project_url', 'publication_url', 'bibtex', 'team_name', 'organization_or_affiliation']
            submission_metadata_file_name = "%s - %s method.txt" % (username_or_team_name, submission.submission_number)
            submission_metadata_file_string = "\n".join(["%s: %s" % (field, getattr(submission, field)) for field in metadata_fields])
            zip_file.writestr(submission_metadata_file_name, submission_metadata_file_string.encode('utf-8'))

        zip_file.close()

        resp = HttpResponse(zip_buffer.getvalue(), mimetype = "application/x-zip-compressed")
        resp['Content-Disposition'] = 'attachment; filename=%s-%s-results.zip' % (competition.title, competition.pk)
        return resp
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print "*** print_tb:"
        traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
        print "*** print_exception:"
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  limit=2, file=sys.stdout)
        print "*** print_exc:"
        traceback.print_exc()
        print "*** format_exc, first and last line:"
        formatted_lines = traceback.format_exc().splitlines()
        msg = "There was an error retrieving the file. Please try again later or report the issue."
        return HttpResponse(msg, status=400, content_type='text/plain')


@login_required
def submission_update_description(request, submission_pk):
    """
    Updates the description of submission.

    :param submission_pk: Submission's primary key.

    """
    try:
        submission = models.CompetitionSubmission.objects.get(pk=submission_pk)
        if submission.participant.user != request.user:
            raise Http404()
        submission.description = request.POST.get('updated_description')
        submission.save()
        return HttpResponse()
    except models.CompetitionSubmission.DoesNotExist:
        raise Http404()


@login_required
def submission_mark_as_failed(request, submission_pk):
    """
    Marks a submission as failed.

    :param submission_pk: Submission's primary key.

    """
    if request.method == "POST":
        try:
            submission = models.CompetitionSubmission.objects.get(pk=submission_pk)
            competition = submission.phase.competition
            if request.user.id != competition.creator_id and request.user not in competition.admins.all():
                raise Http404()
            submission.status = models.CompetitionSubmissionStatus.objects.get(codename="failed")
            submission.save()
            return HttpResponse()
        except models.CompetitionSubmission.DoesNotExist:
            raise Http404()
    raise Http404()


@login_required
def submission_toggle_leaderboard(request, submission_pk):
    """
    Push a submission to the Leaderboard.

    :param submission_pk: Submission's primary key.
    """
    if request.method == "POST":
        try:
            submission = models.CompetitionSubmission.objects.get(pk=submission_pk)
            competition = submission.phase.competition
            if request.user.id != competition.creator_id and request.user not in competition.admins.all():
                raise Http404()

            if submission.status.codename != "finished":
                return HttpResponse(status=400)

            is_on_leaderboard = models.PhaseLeaderBoardEntry.objects.filter(result=submission).exists()
            # If the submission isn't already on the leaderboard, then add it to it
            # otherwise delete it and other submissions else from leaderboard
            if not is_on_leaderboard:
                models.add_submission_to_leaderboard(submission)
            else:
                entries = models.PhaseLeaderBoardEntry.objects.filter(
                    board__phase=submission.phase,
                    result__participant=submission.participant
                )
                for entry in entries:
                    entry.delete()

            return HttpResponse()
        except models.CompetitionSubmission.DoesNotExist:
            raise Http404()
    raise Http404()


@login_required
def submission_re_run(request, submission_pk):
    """
    Allows to re-submit a submission.

    :param submission_pk: Submission's primary key.
    """
    if request.method == "POST":
        try:
            submission = models.CompetitionSubmission.objects.get(pk=submission_pk)
            competition = submission.phase.competition
            if request.user.id != competition.creator_id and request.user not in competition.admins.all():
                raise Http404()

            if settings.USE_AWS:
                file_kwarg = {'s3_file': submission.s3_file}
            else:
                file_kwarg = {'file': submission.file}

            new_submission = models.CompetitionSubmission(
                participant=submission.participant,
                phase=submission.phase,
                docker_image=submission.docker_image,
                **file_kwarg
            )
            new_submission.save(ignore_submission_limits=True)

            evaluate_submission.apply_async((new_submission.pk, submission.phase.is_scoring_only))

            return HttpResponse()
        except models.CompetitionSubmission.DoesNotExist:
            raise Http404()
    raise Http404()


@login_required
def submission_re_run_all(request, phase_pk):
    """Re-runs all submissions in a phase, distinct on the file name, so a submission
    won't be re-ran multiple times if this feature is used repeatedly."""
    if request.method == "POST":
        try:
            phase = models.CompetitionPhase.objects.get(id=phase_pk)
            competition = phase.competition

            if request.user.id != competition.creator_id and request.user not in competition.admins.all():
                raise Http404()

            re_run_all_submissions_in_phase.apply_async((phase_pk,))

            return HttpResponse()
        except models.CompetitionSubmission.DoesNotExist:
            raise Http404()
    raise Http404()


@login_required
def submission_migrate(request, pk):
    '''
    Allow to migrate to submissions manually to next phase.

    :param submission_pk: Submission's primary key.
    '''
    if request.method == "POST":
        try:
            submission = models.CompetitionSubmission.objects.get(pk=pk)
            competition = submission.phase.competition
            if request.user.id != competition.creator.id and request.user not in competition.admins.all():
                raise Http404()

            current_phase_phasenumber = submission.phase.phasenumber
            next_phase = competition.phases.get(phasenumber=current_phase_phasenumber+1)

            new_submission = models.CompetitionSubmission(
                participant=submission.participant,
                file=submission.file,
                phase=next_phase,
                docker_image=submission.docker_image,
            )

            new_submission.save(ignore_submission_limits=True)

            evaluate_submission.apply_async((new_submission.pk, submission.phase.is_scoring_only))
            submission.is_migrated = True
            submission.save()

            return HttpResponse()
        except models.CompetitionSubmission.DoesNotExist:
            raise Http404()
    raise Http404()


@login_required
def competition_dumps_view(request, competition_pk):
    try:
        competition = models.Competition.objects.get(pk=competition_pk)
        if request.user.id != competition.creator_id and request.user not in competition.admins.all():
            raise Http404()
        dumps = competition.dumps.all().order_by('-timestamp')
    except ObjectDoesNotExist:
        raise Http404()

    return render(request, "web/competitions/dumps.html", {"dumps": dumps, "competition": competition})


@login_required
def start_make_bundle_task(request, competition_pk):
    competition = models.Competition.objects.get(pk=competition_pk)
    if request.user != competition.creator and request.user not in competition.admins.all():
        raise Http404()
    # make_modified_bundle.apply_async((competition.pk, dataset_flag,))
    if request.method == "POST":
        exclude_datasets_flag = request.POST.get('exclude_datasets_flag')
        if exclude_datasets_flag == "false":
            exclude_datasets_flag = False
        else:
            exclude_datasets_flag = True
        print("Datasets flag is {}".format(exclude_datasets_flag))
        make_modified_bundle.apply_async((competition.pk, exclude_datasets_flag,))
    return HttpResponse()


class CompetitionDumpDeleteView(DeleteView):
    model = models.CompetitionDump
    template_name = 'web/competitions/delete_dump_confirm.html'

    def dispatch(self, request, *args, **kwargs):
        pk = kwargs.get('pk')
        dump = models.CompetitionDump.objects.get(pk=pk)
        competition = dump.competition
        user = request.user
        if user.id != competition.creator.id and user not in competition.admins.all():
            return HttpResponseForbidden()
        else:
            return super(CompetitionDumpDeleteView, self).dispatch(request, *args, **kwargs)

    def get_success_url(self):
        dump = self.object
        return reverse('competitions:dumps', kwargs={'competition_pk': dump.competition.pk})


## DeveloperLab
di = {}

docfilecopy = ''
ifsendtoBMT = 0
rec_status = 0
rec_url = ''
picture_url = ''
csv_url = ''

@login_required
def developerlab(request):
    # Handle file upload
    global iftrain
    global iftest
    global ifunlabel
    global ifdataset
    global numofdataset
    global numofunlabel
    global ifzip
    global di
    global docfilecopy
    global ifsendtoBMT
    global rec_status
    global rec_url
    global picture_url
    global csv_url
    global bmtcsv_url
    global task_id
    global visual_value

    bmtcsv_url = '/static/img/partpicture/' + str(request.user) + '/bmtresponse.csv'
    visual_value = ['yes','no']
    if request.method == 'POST':

        submit = request.POST.get('uploadfile', None)
        clear = request.POST.get('clearall', None)
        getbmtresult = request.POST.get('getlabeled', None)

        if clear:
            di = {}
            iftrain = 0
            iftest = 0
            ifunlabel = 0
            ifdataset = 0
            numofdataset = 0
            numofunlabel = 0
            ifzip = 0 # 0 is none, 1 is valid, 2 is wrong
            picture_url = '/static/img/partpicture/' + str(request.user) + '/compare.png'
            csv_url = '/static/img/partpicture/' + str(request.user) + '/dict.csv'
            if os.path.exists(picture_url):
                os.remove(picture_url)
            if os.path.exists(csv_url):
                os.remove(csv_url)

            return HttpResponseRedirect(reverse('developerlab_upload'), {'ifzip':ifzip,'iftrain':iftrain,'ifunlabel':ifunlabel,'iftest':iftest, 'ifdataset':ifdataset, 'numofdataset':numofdataset, 'numofunlabel':numofunlabel, 'bmtcsv_url':bmtcsv_url})

        if getbmtresult:
            # task_id = 19
            binary = BinaryClassTest()
            # numneedtobelabeled, trainandtest, testsize, markmethod
            binary.getbmtresult('anbang', '123456', task_id, visual_value)
            return HttpResponseRedirect(reverse('developerlab_upload'),
                                        {'ifzip': ifzip, 'iftrain': iftrain, 'ifunlabel': ifunlabel, 'iftest': iftest,
                                         'ifdataset': ifdataset, 'numofdataset': numofdataset,
                                         'numofunlabel': numofunlabel, 'bmtcsv_url':bmtcsv_url})
#
        if submit:
            di = {}
            form = DocumentForm(request.POST, request.FILES)
            if form.is_valid():

                userpath = '/app/codalab/thirdpart/'+str(request.user) +'/'
                if not os.path.exists(userpath):
                    os.mkdir(userpath)

                docfile = request.FILES['docfile']
                with open('/app/codalab/thirdpart/'+str(request.user)+'/docfile.zip', 'wb+') as destination:
                    for chunk in docfile.chunks():
                        destination.write(chunk)

                if docfile.name.split('/')[-1].split('.')[-1]=='zip':
                    iftrain = 0
                    iftest = 0
                    ifunlabel = 0
                    ifdataset = 0
                    numofdataset = 0
                    numofunlabel = 0

                    ifzip = 1 # 0 is none, 1 is valid, 2 is wrong
                    ziplist = zipfile.ZipFile(docfile)

                    unlabeldataset = list(filter(lambda x: '/' in x , ziplist.namelist()))#
                    numofdataset = len(unlabeldataset)

                    expunlabeldataset = list(filter(lambda x: '/' not in x, ziplist.namelist()))

                    if numofdataset>0:
                        ifdataset = 1
                    if 'train.txt' in expunlabeldataset:
                        iftrain = 1
                    if 'unlabel.txt' in expunlabeldataset:
                        fileunlabel = ziplist.read('unlabel.txt').decode('utf-8')
                        numofunlabel = fileunlabel.count('\n')+1
                        ifunlabel = 1

                    if 'test.txt' in expunlabeldataset:
                        iftest = 1
                    if iftrain==1 and ifunlabel==1:
                        newdoc = Document(docfile = request.FILES['docfile'])
                        newdoc.creator = request.user
                        # newdoc.()#[todo]:change to savesave

                else:
                    ifzip = 2

                # Redirect to the document list after POST
                return HttpResponseRedirect(reverse('developerlab_upload'),{'ifzip':ifzip,'iftrain':iftrain,'ifunlabel':ifunlabel,'iftest':iftest, 'ifdataset':ifdataset, 'numofdataset':numofdataset, 'numofunlabel':numofunlabel, 'bmtcsv_url':bmtcsv_url})
        else:

            #doclist = list(Document.objects.filter(creator = request.user))
            #docfilecopy = Document.objects.filter(creator = request.user)[len(doclist)-1]['docfile']
            docfilecopy = '/app/codalab/thirdpart/'+str(request.user)+'/docfile.zip'
            documentss = Document.objects.all()
            documentss = documentss.filter(creator=request.user)

            if request.method == 'GET':
                return HttpResponseRedirect(reverse('developerlab_upload'),{'documentss':documentss,'di':di,'docfilecopy':docfilecopy})
            elif request.method == 'POST':
                di = {}
                ifsenttoBMT = 0
                kind = request.POST.get('kind')#text,image
                model = request.POST.get('model')#logistic, svm
                strategy = request.POST.get('strategy')#binary,multiclass,multilabel
                alg = request.POST.get('alg')#us,qbc

                bmtpassword = request.POST.get('bmtpassword')
                bmttask = {}
                bmttask['name'] = str(request.POST.get('bmt_taskname'))
                bmttask['describe'] = str(request.POST.get('bmt_taskdescribe'))
                bmttask['price'] = int(request.POST.get('bmt_price'))
                bmttask['deadline'] = str(request.POST.get('bmt_deadline')) + '000000'

                # testsize = request.POST.get('testsize')
                batchsize = request.POST.get('batchsize')
                numneedtobelabeled = int(request.POST.get('numneedtobelabeled'))
                markmethod = request.POST.get('markmethod')#returnentities, sendtoBMT

                username = str(request.user)
                useremail = str(request.user.email)
                #userpassword = str(request.user.password)


                if batchsize:
                    batchsize = int(batchsize)
                else:
                    batchsize = 1
                if markmethod == 'returnentities':
                    markmethod = 1
                    ifsendtoBMT = 0
                    binary = BinaryClassTest()
                    #numneedtobelabeled, trainandtest, testsize,markmethod
                    di, csvdir = binary.maintodo(kind, model, strategy, alg, numneedtobelabeled, 1, batchsize, markmethod, docfilecopy, username, useremail, bmtpassword, bmttask)

                    picture_url = '/static/img/partpicture/'+str(request.user)+'/compare.png'
                    csv_url = '/static/img/partpicture/'+str(request.user)+'/dict.csv'
                    return HttpResponseRedirect(reverse('developerlab_upload'), {'di': di, 'docfilecopy': docfilecopy, 'kind': kind, 'picture_url': picture_url, 'csvdir': csv_url, 'bmtcsv_url':bmtcsv_url})

                #[TODO]向标注平台发送
                else:
                    markmethod = 0
                    ifsendtoBMT = 1
                    binary = BinaryClassTest()
                    #numneedtobelabeled, trainandtest, testsize, markmethod
                    rec_status, rec_url, di, task_id, visual_value = binary.maintodo(kind, model, strategy, alg, numneedtobelabeled, 1, batchsize, markmethod, docfilecopy, username, useremail, bmtpassword, bmttask)
                    picture_url = '/static/img/partpicture/'+str(request.user)+'/compare.png'
                    return HttpResponseRedirect(reverse('developerlab_upload'), {'di':di,'docfilecopy':docfilecopy,'ifsendtoBMT':ifsendtoBMT,'rec_status':rec_status, 'rec_url':rec_url,'picture_url':picture_url , 'bmtcsv_url':bmtcsv_url})
            else:

                return HttpResponseRedirect(reverse('developerlab_upload'), {'docfilecopy':docfilecopy})

            return HttpResponseRedirect(reverse('developerlab_upload'), {'docfilecopy':docfilecopy})


    else:
        form = DocumentForm() # A empty, unbound form

    # Load documents for the list page
    #documentss = Document.objects.all()

    doclist = list(Document.objects.filter(creator=request.user))
    documentss = Document.objects.filter(creator=request.user)
    if(len(doclist)>10):
        documentss = documentss.filter(creator = request.user)[len(doclist)-10:len(doclist)]
    else:
        documentss = documentss.filter(creator=request.user)
    return render_to_response(
        'web/my/developerlab.html',
        {'documentss': documentss, 'form': form, 'ifzip': ifzip,'iftrain': iftrain, 'ifunlabel': ifunlabel, 'iftest': iftest, 'ifdataset': ifdataset,'ifsendtoBMT': ifsendtoBMT,
         'numofdataset': numofdataset, 'numofunlabel': numofunlabel, 'di': di,
         'rec_status': rec_status, 'rec_url': rec_url, 'picture_url': picture_url,'csvdir': csv_url, 'bmtcsv_url':bmtcsv_url
         },
        context_instance = RequestContext(request)
    )


class CompetitionWidgetsView(LoginRequiredMixin, DetailView):
    queryset = models.Competition.objects.all()
    template_name = 'web/competitions/widgets.html'

    def get_object(self, queryset=None):
        self.object = super(CompetitionWidgetsView, self).get_object()
        if self.request.user != self.object.creator and self.request.user not in self.object.admins.all():
            raise Http404()
        return self.object


class WidgetMixin(object):

    def get_context_data(self, **kwargs):
        context = super(WidgetMixin, self).get_context_data(**kwargs)
        context['current_phase'] = get_current_phase(self.object)
        return context


class CompetitionSubmissionWidgetView(WidgetMixin, DetailView):
    queryset = models.Competition.objects.all()
    template_name = 'web/widget_iframes/submission.html'

    def get_context_data(self, **kwargs):
        context = super(CompetitionSubmissionWidgetView, self).get_context_data(**kwargs)
        context['phase'] = get_current_phase(self.object)
        phase = context['phase']
        competition = self.object

        if settings.USE_AWS:
            context['form'] = forms.SubmissionS3UploadForm

        if competition.participants.filter(user__in=[self.request.user]).exists():
            participant = competition.participants.get(user=self.request.user)
            if participant.status.codename == models.ParticipantStatus.APPROVED:
                submissions = models.CompetitionSubmission.objects.filter(
                    participant=participant,
                    phase=phase
                ).select_related('status').order_by('submitted_at')

                # find which submission is in the leaderboard, if any and only if phase allows seeing results.
                id_of_submission_in_leaderboard = -1
                if not phase.is_blind:
                    leaderboard_entry = models.PhaseLeaderBoardEntry.objects.filter(
                        board__phase=phase,
                        result__participant__user=self.request.user
                    ).select_related('result', 'result__participant')
                    if leaderboard_entry:
                        id_of_submission_in_leaderboard = leaderboard_entry[0].result.pk
                submission_info_list = []
                for submission in submissions:
                    try:
                        default_score = float(submission.get_default_score())
                    except (TypeError, ValueError):
                        default_score = '---'
                    submission_info = {
                        'id': submission.id,
                        'number': submission.submission_number,
                        'filename': submission.get_filename(),  # left as call for legacy update of readable_filename on subs.
                        'submitted_at': submission.submitted_at,
                        'status_name': submission.status.name,
                        'is_finished': submission.status.codename == 'finished',
                        'is_in_leaderboard': submission.id == id_of_submission_in_leaderboard,
                        'exception_details': submission.exception_details,
                        'description': submission.description,
                        'team_name': submission.team_name,
                        'method_name': submission.method_name,
                        'method_description': submission.method_description,
                        'project_url': submission.project_url,
                        'publication_url': submission.publication_url,
                        'bibtex': submission.bibtex,
                        'organization_or_affiliation': submission.organization_or_affiliation,
                        'is_public': submission.is_public,
                        'score': default_score,
                    }
                    submission_info_list.append(submission_info)
                context['submission_info_list'] = submission_info_list
                context['phase'] = phase

        try:
            last_submission = models.CompetitionSubmission.objects.filter(
                participant__user=self.request.user,
                phase=context['phase']
            ).latest('submitted_at')
            context['last_submission_team_name'] = last_submission.team_name
            context['last_submission_method_name'] = last_submission.method_name
            context['last_submission_method_description'] = last_submission.method_description
            context['last_submission_project_url'] = last_submission.project_url
            context['last_submission_publication_url'] = last_submission.publication_url
            context['last_submission_bibtex'] = last_submission.bibtex
            context['last_submission_organization_or_affiliation'] = last_submission.organization_or_affiliation
        except ObjectDoesNotExist:
            pass
        return context


class CompetitionLeaderboardWidgetView(WidgetMixin, DetailView):
    queryset = models.Competition.objects.all()
    template_name = 'web/widget_iframes/leaderboard.html'

    def get_context_data(self, **kwargs):
        context = super(CompetitionLeaderboardWidgetView, self).get_context_data(**kwargs)
        try:
            context['block_leaderboard_view'] = True
            competition = self.object
            phase = get_current_phase(competition)
            is_owner = self.request.user.id == competition.creator_id
            context['competition_admins'] = competition.admins.all()
            context['is_owner'] = is_owner
            context['phase'] = phase
            context['groups'] = phase.scores()

            for group in context['groups']:
                for _, scoredata in group['scores']:
                    sub = models.CompetitionSubmission.objects.get(pk=scoredata['id'])
                    scoredata['date'] = sub.submitted_at
                    scoredata['count'] = sub.phase.submissions.filter(participant=sub.participant).count()
                    if sub.team:
                        scoredata['team_name'] = sub.team.name

            user = self.request.user

            # Will allow creator and admin to see Leaderboard in advanced
            if user == phase.competition.creator or user in phase.competition.admins.all():
                context['block_leaderboard_view'] = False
        except:
            context['error'] = traceback.format_exc()
        finally:
            return context


@login_required
def user_lookup(request):
    search = request.GET.get('term', '')
    filters = Q()
    show_emails = False

    if search:
        filters |= Q(username__icontains=search)
    if request.user.is_superuser or request.user.is_staff:
        show_emails = True
        filters |= Q(email__icontains=search)

    users = ClUser.objects.filter(filters)[:5]

    # Helper to print username with email for admins
    def _get_username(user):
        if show_emails:
            return "{} ({})".format(user.username, user.email)
        else:
            return user.username

    return HttpResponse(
        json.dumps({"results": [
            {"id": u.pk, "text": _get_username(u)} for u in users]
        }),
        content_type="application/json"
    )

