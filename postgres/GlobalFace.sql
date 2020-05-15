/*
Tabla que contiene los registros individuales de cada persona.
ID_GLOBAL será utilizado para el entrenamiento de los clasificadores.
*/
drop table if exists registros;
create table registros(
  id_registro bigserial primary key,
  id_global bigint,
  nombre varchar,
  comentarios varchar,
  id_tienda bigint,
  id_lista bigint,
  fecha_registro timestamp
);

/*
Tabla que contiene las características faciales de cada imagen original y
aumentada para cada persona registrada

--> imagen:
    Imagen codificada en base 64
--> features:
    Vector de 128 números flotantes que representan un rostro
--> aumentado:
    Verdadero si la imagen fue generada mediante aumento de datos, Falso si es
    original
*/
drop table if exists features;
create table features(
  id_features bigserial primary key,
  id_registro bigint,
  id_global bigint,
  imagen varchar,
  features double precision[],
  aumentado bool
);

/*
Tabla que contiene las listas con las que se registrarán a las personas
*/
drop table if exists listas;
create table listas(
  id_lista bigserial primary key,
  nombre_lista varchar,
  descripcion varchar,
  fecha_actualizacion timestamp
);

insert into listas
values(default, 'Lista Blanca', 'Lista de personas de confianza', now()::timestamp),
      (default, 'Lista Negra', 'Lista de personas buscadas', now()::timestamp);

/*
Tabla que contiene las get_tiendas en las que se registrarán a las personas
*/
drop table if exists tiendas;
create table tiendas(
  id_tienda bigserial primary key,
  nombre_tienda varchar
);

insert into tiendas
values(default,'Plaza Carso'),
      (default,'Plaza Inbursa'),
      (default,'Toreo');

/*
Función que inserta registros en la base de datos
*/
drop function if exists put_registros;
create function put_registros(
  glb bigint,
  nom varchar,
  com varchar,
  iti bigint,
  lis bigint,
  out idr bigint
) as
$$
begin
  insert into registros
  values(default, glb, nom, com, iti, lis, now()::timestamp)
  returning id_registro
  into idr;
end
$$
language plpgsql;

/*
Función que inserta features en la base de datos
*/
drop function if exists put_features;
create function put_features(
  idr bigint,
  glb bigint,
  img varchar,
  fts double precision[],
  aum bool,
  out idf bigint
) as
$$
begin
  insert into features
  values(default, idr, glb, img, fts, aum)
  returning id_features
  into idf;
end
$$
language plpgsql;

/*
Función que extrae los registros de la base de datos
*/
drop function if exists get_registros;
create function get_registros()
returns table(
  idg bigint,
  nom varchar,
  com varchar,
  iti bigint,
  lis bigint,
  fch varchar
) as
$$
begin
  return query
    select
      id_global,
      nombre,
      comentarios,
      id_tienda,
      id_lista,
      fecha_registro::varchar
    from registros;
end
$$
language plpgsql;

/*
Función que extrae una imagen original de un registro con id
*/
drop function if exists get_mug;
create function get_mug(
  idg bigint,
  out img varchar
) as
$$
begin
  select imagen into img
  from features
  where
    id_global = idg
  and
    not aumentado
  limit 1;
end
$$
language plpgsql;


/*
Función que elimina un registro y sus features egún su id
*/
drop function if exists delete_registros;
create function delete_registros(
  idg bigint
) returns void as
$$
begin
  delete from registros
  where id_global = idg;

  delete from features
  where id_global = idg;
end
$$
language plpgsql;


/*
Función que extrae las listas
*/
drop function if exists get_listas;
create function get_listas()
returns table(ili bigint, nli varchar, des varchar, act varchar) as
$$
begin
  return query
    select id_lista, nombre_lista, descripcion, fecha_actualizacion::varchar
    from listas;
end
$$
language plpgsql;

/*
Función que extrae las tiendas
*/
drop function if exists get_tiendas;
create function get_tiendas()
returns table(iti bigint, nti varchar) as
$$
begin
  return query
    select id_tienda, nombre_tienda
    from tiendas;
end
$$
language plpgsql;

/*
Función que obtiene el siguiente id_global a utilizar
*/
drop function if exists next_id_global;
create function next_id_global(
  out nxt bigint
) as
$$
begin
  select max(id_global) + 1 into nxt
  from registros;
end
$$
language plpgsql;

/*
Función que verifica si existe un id_global
*/
drop function if exists id_global_exists;
create function id_global_exists(
  idg bigint,
  out exi bool
) as
$$
begin
  select exists(
    select 1
    from registros
    where id_global = idg
    limit 1
  ) into exi;
end
$$
language plpgsql;

/*
Extensión utilizada para operaciones de similitud de strings
*/
drop extension if exists pg_trgm;
create extension pg_trgm;

/*
Función que devuelve registros con nombres similares al proporcionado
*/
drop function if exists similar_names;
create function similar_names(
  nam varchar
) returns table(
  nom varchar,
  idg bigint,
  com varchar,
  iti bigint,
  ili bigint
) as
$$
begin
  return query
    select distinct(nombre), id_global, comentarios, id_tienda, lista
    from registros
    where similarity(nombre, nam) >= 0.5;
end
$$
language plpgsql;

/*
Función que extrae los features de la base de datos
*/
drop function if exists get_features;
create function get_features()
returns table(
  idg bigint,
  fts double precision[]
) as
$$
begin
  return query
    select id_global, features
    from features;
end
$$
language plpgsql;

/*
Función que extrae el nombre de un registro según su id
*/
drop function if exists name_from_id;
create function name_from_id(
  idp bigint,
  out nap varchar
) as
$$
begin
  select nombre into nap
  from registros where id_global = idp;
end
$$
language plpgsql;
