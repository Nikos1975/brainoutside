from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("events", "0003_event_type_error")]  # noqa: RUF012

    operations = [  # noqa: RUF012
        migrations.AddField(
            model_name="sdkoperation",
            name="reserved_cost_usd",
            field=models.DecimalField(
                decimal_places=6, default=0, max_digits=10
            ),
        ),
        migrations.AddField(
            model_name="sdkoperation",
            name="cost_source",
            field=models.CharField(blank=True, default="", max_length=16),
        ),
    ]
